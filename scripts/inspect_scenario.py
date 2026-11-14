"""Khảo sát XML và chạy SUMO với chương trình đèn gốc, không điều khiển pha."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import sumolib
import traci


ROOT = Path(__file__).resolve().parents[1]


def hash_inputs(scenario):
    """Nhận tên kịch bản, trả SHA-256 của bốn tệp dữ liệu gốc để đối chiếu."""
    folder = ROOT / "data" / scenario
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.iterdir()) if path.is_file()
    }


def inspect_xml(scenario):
    """Đọc cấu hình, mạng, nhu cầu; trả thông tin pha (giây), làn và trips (xe)."""
    folder = ROOT / "data" / scenario
    config = ET.parse(folder / f"{scenario}.sumocfg").getroot()
    net_path = folder / config.find("input/net-file").get("value")
    route_path = folder / config.find("input/route-files").get("value")
    network = ET.parse(net_path).getroot()
    demand = ET.parse(route_path).getroot()
    trips = demand.findall("trip")
    departures = [float(trip.get("depart")) for trip in trips]
    begin = float(config.find("time/begin").get("value"))
    end = float(config.find("time/end").get("value"))
    connections = []
    for connection in network.findall("connection"):
        if connection.get("tl") is not None:
            item = dict(connection.attrib)
            item["incoming_lane"] = f"{item['from']}_{item['fromLane']}"
            item["outgoing_lane"] = f"{item['to']}_{item['toLane']}"
            connections.append(item)
    programs = []
    for logic in network.findall("tlLogic"):
        phases = [dict(phase.attrib) for phase in logic.findall("phase")]
        programs.append({
            **logic.attrib,
            "phases": phases,
            "cycle_seconds": sum(float(phase["duration"]) for phase in phases),
            "has_yellow": any("y" in phase["state"].lower() for phase in phases),
            "has_all_red": any(set(phase["state"].lower()) == {"r"} for phase in phases),
        })
    # Giữ nguyên bitset để giai đoạn 2 kiểm chứng xung đột; không suy ra từ index pha.
    junctions = [
        {**node.attrib, "requests": [dict(row.attrib) for row in node.findall("request")]}
        for node in network.findall("junction") if node.findall("request")
    ]
    return {
        "scenario": scenario, "config": str(folder / f"{scenario}.sumocfg"),
        "begin_seconds": begin, "end_seconds": end, "duration_seconds": end - begin,
        "network_version": network.get("version"),
        "external_edges": len([e for e in network.findall("edge") if e.get("function") != "internal"]),
        "internal_edges": len([e for e in network.findall("edge") if e.get("function") == "internal"]),
        "programs": programs, "controlled_connections": connections, "junctions": junctions,
        "demand": {
            "tags": dict(Counter(element.tag for element in demand)),
            "trip_count": len(trips), "unique_trip_ids": len({t.get("id") for t in trips}),
            "first_depart_seconds": min(departures), "last_depart_seconds": max(departures),
            "trips_in_window": sum(begin <= value < end for value in departures),
            "vehicle_types": [dict(t.attrib) for t in demand.findall("vType")],
            "trip_types": dict(Counter(t.get("type", "DEFAULT_VEHTYPE") for t in trips)),
            "origin_destination": [
                {"from": pair[0], "to": pair[1], "count": count}
                for pair, count in sorted(Counter((t.get("from"), t.get("to")) for t in trips).items())
            ],
        },
    }


def inspect_tls(connection, survey):
    """Đọc TLS qua TraCI; đối chiếu pha và từng incoming→outgoing→via với XML."""
    records = []
    for tls_id in connection.trafficlight.getIDList():
        lanes = list(connection.trafficlight.getControlledLanes(tls_id))
        links = connection.trafficlight.getControlledLinks(tls_id)
        programs = []
        for logic in connection.trafficlight.getAllProgramLogics(tls_id):
            phases = [{"duration": p.duration, "state": p.state} for p in logic.phases]
            xml_logic = next(p for p in survey["programs"]
                             if p["id"] == tls_id and p["programID"] == logic.programID)
            expected = [{"duration": float(p["duration"]), "state": p["state"]}
                        for p in xml_logic["phases"]]
            if phases != expected:
                raise ValueError(f"Pha TraCI khác XML: {tls_id}")
            programs.append({"program_id": logic.programID, "phases": phases})
        observed = {
            (index, item[0], item[1], item[2])
            for index, group in enumerate(links) for item in group
        }
        expected = {
            (int(item["linkIndex"]), item["incoming_lane"], item["outgoing_lane"], item["via"])
            for item in survey["controlled_connections"] if item["tl"] == tls_id
        }
        if observed != expected:
            raise ValueError(f"Controlled links TraCI khác XML: {tls_id}")
        records.append({
            "id": tls_id, "program_id": connection.trafficlight.getProgram(tls_id),
            "controlled_lanes_raw": lanes, "controlled_lanes_unique": sorted(set(lanes)),
            "controlled_links": links, "programs": programs,
            "xml_matches_traci": True,
        })
    if {r["id"] for r in records} != {p["id"] for p in survey["programs"]}:
        raise ValueError("Danh sách TLS TraCI khác XML")
    return records


def run_original(survey, output, gui=False, steps=None, delay=0):
    """Chạy chương trình gốc; ghi số xe và quan sát pha mỗi simulation step, đơn vị giây."""
    binary = shutil.which("sumo-gui" if gui else "sumo")
    if binary is None:
        raise FileNotFoundError("Không tìm thấy SUMO trong PATH")
    command = [binary, "-c", survey["config"], "--no-step-log", "true",
               "--duration-log.statistics", "true",
               "--log", str(output / "sumo.log"),
               "--error-log", str(output / "warnings.log"),
               "--summary-output", str(output / "summary.xml"),
               "--tripinfo-output", str(output / "tripinfo.xml"),
               "--tripinfo-output.write-unfinished", "true",
               "--tripinfo-output.write-undeparted", "true"]
    if gui:
        # Tự đóng GUI khi TraCI kết thúc, tránh close() chờ hộp thoại người dùng.
        command += ["--start", "--quit-on-end", "--delay", str(delay)]
    # Không override seed, begin/end, step-length hay bất kỳ tham số điều khiển nào.
    traci.start(command)
    connection = traci.getConnection()
    result = {"command": command, "version": connection.getVersion(), "steps": 0,
              "departed": 0, "arrived": 0, "teleport_starts": 0,
              "collision_vehicle_events": 0}
    try:
        result["tls"] = inspect_tls(connection, survey)
        result["options"] = {key: connection.simulation.getOption(key) for key in
                             ("seed", "random", "step-length", "begin", "end", "collision.action",
                              "time-to-teleport", "route-steps", "xml-validation")}
        result["initial_time_seconds"] = connection.simulation.getTime()
        with (output / "phase_observations.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["observed_time_s", "tls_id", "phase_index", "state", "next_switch_s"])
            while True:
                time = connection.simulation.getTime()
                for record in result["tls"]:
                    tls_id = record["id"]
                    writer.writerow([time, tls_id, connection.trafficlight.getPhase(tls_id),
                                     connection.trafficlight.getRedYellowGreenState(tls_id),
                                     connection.trafficlight.getNextSwitch(tls_id)])
                if time >= survey["end_seconds"] or (steps is not None and result["steps"] >= steps):
                    break
                connection.simulationStep()
                result["steps"] += 1
                result["departed"] += connection.simulation.getDepartedNumber()
                result["arrived"] += connection.simulation.getArrivedNumber()
                result["teleport_starts"] += connection.simulation.getStartingTeleportNumber()
                result["collision_vehicle_events"] += connection.simulation.getCollidingVehiclesNumber()
        result["final_time_seconds"] = connection.simulation.getTime()
        result["active_at_end"] = connection.vehicle.getIDCount()
        result["pending_at_end"] = len(connection.simulation.getPendingVehicles())
        result["completed_configured_window"] = time >= survey["end_seconds"]
    finally:
        connection.close()
    # Chỉ đếm/đối chiếu phục vụ khảo sát; chưa triển khai bộ metrics dùng chung.
    trips = ET.parse(output / "tripinfo.xml").getroot().findall("tripinfo")
    result["tripinfo_counts"] = {
        "arrived": sum(float(t.get("arrival")) >= 0 for t in trips),
        "unfinished_inserted": sum(float(t.get("arrival")) < 0 and float(t.get("depart")) >= 0 for t in trips),
        "undeparted": sum(float(t.get("depart")) < 0 for t in trips),
    }
    return result


def main():
    """Đọc CLI, khảo sát một kịch bản và lưu bằng chứng vào results/ (không ghi vào data/)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("cologne1", "ingolstadt1"))
    parser.add_argument("--run", action="store_true", help="Chạy SUMO nguyên trạng qua TraCI")
    parser.add_argument("--gui", action="store_true", help="Chạy SUMO-GUI qua TraCI")
    parser.add_argument("--steps", type=int, help="Chỉ smoke test N bước; mặc định chạy hết thời gian gốc")
    parser.add_argument("--delay", type=int, default=0, help="Độ trễ hiển thị GUI (ms), không đổi simulation step")
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps phải > 0")
    if args.steps is not None and not (args.run or args.gui):
        parser.error("--steps cần --run hoặc --gui")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    mode = "gui" if args.gui else "sumo" if args.run else "xml"
    output = ROOT / "results" / "stage1" / f"{args.scenario}_{mode}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    survey = inspect_xml(args.scenario)
    survey["recorded_at_utc"] = stamp
    survey["input_sha256_before"] = hash_inputs(args.scenario)
    survey["inspector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    survey["environment"] = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "SUMO_HOME": os.environ.get("SUMO_HOME"), "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "traci_path": traci.__file__, "sumolib_path": sumolib.__file__,
        "installed_distributions": {name: version(name) for name in ("traci", "sumolib")},
        "sumo_binary": shutil.which("sumo"), "sumo_gui_binary": shutil.which("sumo-gui"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    }
    if args.run or args.gui:
        survey["run"] = run_original(survey, output, args.gui, args.steps, args.delay)
    survey["input_sha256_after"] = hash_inputs(args.scenario)
    if survey["input_sha256_before"] != survey["input_sha256_after"]:
        raise RuntimeError("Dữ liệu gốc đã thay đổi trong lúc khảo sát")
    (output / "survey.json").write_text(json.dumps(survey, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Survey: {output / 'survey.json'}")
    print(f"Trips: {survey['demand']['trip_count']}; window: {survey['duration_seconds']} s")
    if "run" in survey:
        result = survey["run"]
        print(f"Steps: {result['steps']}; departed: {result['departed']}; arrived: {result['arrived']}; "
              f"active: {result['active_at_end']}; pending: {result['pending_at_end']}")


if __name__ == "__main__":
    main()
