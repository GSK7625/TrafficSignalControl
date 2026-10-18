"""Module for inspecting RESCO SUMO dataset files."""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


@dataclass
class PhaseInfo:
    """Information about a traffic light phase."""
    index: int
    duration: float
    state: str
    min_dur: Optional[float] = None
    max_dur: Optional[float] = None


@dataclass
class ScenarioInfo:
    """Information regarding a SUMO scenario."""
    name: str
    cfg_path: str
    net_path: str
    rou_path: str
    begin_time: float
    end_time: float
    duration: float
    tls_count: int
    tls_ids: List[str] = field(default_factory=list)
    phases_per_tls: Dict[str, List[PhaseInfo]] = field(default_factory=dict)
    controlled_lanes_per_tls: Dict[str, List[str]] = field(default_factory=dict)
    controlled_links_per_tls: Dict[str, int] = field(default_factory=dict)
    trips_or_vehicles_count: int = 0
    vehicle_depart_range: Optional[Tuple[float, float]] = None
    auxiliary_files: List[str] = field(default_factory=list)
    rou_files: List[str] = field(default_factory=list)


def inspect_scenario(scenario_name: str, data_dir: str = "data") -> ScenarioInfo:
    """Inspects a scenario directory and extracts metadata."""
    scenario_path = os.path.join(data_dir, scenario_name)
    if not os.path.isdir(scenario_path):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_path}")

    cfg_file = os.path.join(scenario_path, f"{scenario_name}.sumocfg")
    if not os.path.isfile(cfg_file):
        candidates = [f for f in os.listdir(scenario_path) if f.endswith(".sumocfg")]
        cfg_file = os.path.join(scenario_path, candidates[0]) if candidates else cfg_file

    cfg_root = ET.parse(cfg_file).getroot()
    net_node = cfg_root.find(".//net-file")
    net_file = os.path.normpath(os.path.join(scenario_path, net_node.get("value", "")))

    rou_node = cfg_root.find(".//route-files")
    rou_files = [
        os.path.normpath(os.path.join(scenario_path, p.strip()))
        for p in (rou_node.get("value", "").split(",") if rou_node is not None else [])
        if p.strip()
    ]
    rou_file = rou_files[0] if rou_files else ""

    begin_node = cfg_root.find(".//begin")
    begin_time = float(begin_node.get("value", 0)) if begin_node is not None else 0.0
    end_node = cfg_root.find(".//end")
    end_time = float(end_node.get("value", 3600)) if end_node is not None else 3600.0

    net_root = ET.parse(net_file).getroot()
    tl_nodes = net_root.findall("tlLogic")
    tls_ids = [tl.get("id") for tl in tl_nodes if tl.get("id")]

    phases_per_tls: Dict[str, List[PhaseInfo]] = {
        tl.get("id"): [
            PhaseInfo(
                index=idx,
                duration=float(p.get("duration", 0)),
                state=p.get("state", ""),
                min_dur=float(p.get("minDur")) if p.get("minDur") is not None else None,
                max_dur=float(p.get("maxDur")) if p.get("maxDur") is not None else None,
            )
            for idx, p in enumerate(tl.findall("phase"))
        ]
        for tl in tl_nodes
    }

    controlled_lanes_per_tls: Dict[str, List[str]] = {}
    controlled_links_per_tls: Dict[str, int] = {}
    for tl_id in tls_ids:
        connections = net_root.findall(f".//connection[@tl='{tl_id}']")
        phases = phases_per_tls.get(tl_id, [])
        controlled_links_per_tls[tl_id] = len(phases[0].state) if phases and phases[0].state else len(connections)
        controlled_lanes_per_tls[tl_id] = sorted(list({
            f"{c.get('from')}_{c.get('fromLane')}"
            for c in connections
            if c.get("from") and c.get("fromLane") is not None
        }))

    # Demand count and departure range
    all_departs: List[float] = []
    demand_count = 0
    for r_file in rou_files:
        if os.path.isfile(r_file):
            rou_root = ET.parse(r_file).getroot()
            items = rou_root.findall("trip") + rou_root.findall("vehicle") + rou_root.findall("flow")
            demand_count += len(items)
            for item in items:
                d = item.get("depart") or item.get("begin")
                if d is not None:
                    try:
                        all_departs.append(float(d))
                    except ValueError:
                        pass

    depart_range = (min(all_departs), max(all_departs)) if all_departs else None
    primary_files = {os.path.basename(cfg_file), os.path.basename(net_file)} | {os.path.basename(f) for f in rou_files}
    aux = [f for f in os.listdir(scenario_path) if f not in primary_files]

    return ScenarioInfo(
        name=scenario_name,
        cfg_path=cfg_file,
        net_path=net_file,
        rou_path=rou_file,
        begin_time=begin_time,
        end_time=end_time,
        duration=end_time - begin_time,
        tls_count=len(tls_ids),
        tls_ids=tls_ids,
        phases_per_tls=phases_per_tls,
        controlled_lanes_per_tls=controlled_lanes_per_tls,
        controlled_links_per_tls=controlled_links_per_tls,
        trips_or_vehicles_count=demand_count,
        vehicle_depart_range=depart_range,
        auxiliary_files=aux,
        rou_files=rou_files,
    )


def get_window_demand(
    scenario_name_or_info: Any,
    window_begin: float,
    window_end: float,
    data_dir: str = "data",
) -> Dict[str, Any]:
    """Calculates scheduled traffic demand within [window_begin, window_end)."""
    info = scenario_name_or_info if isinstance(scenario_name_or_info, ScenarioInfo) else inspect_scenario(scenario_name_or_info, data_dir)
    rou_files = info.rou_files or ([info.rou_path] if info.rou_path else [])
    count, total = 0, 0
    for r in rou_files:
        if os.path.isfile(r):
            for elem in ET.parse(r).getroot().findall("*"):
                if elem.tag in ("trip", "vehicle", "flow"):
                    total += 1
                    d = elem.get("depart") or elem.get("begin")
                    if d is not None:
                        try:
                            if window_begin <= float(d) < window_end:
                                count += 1
                        except ValueError:
                            pass
    return {
        "scheduled_demand_in_window": count,
        "demand_counting_mode": "exact",
        "window_begin": float(window_begin),
        "window_end": float(window_end),
        "window_duration": max(0.0, float(window_end) - float(window_begin)),
        "total_route_demand": total,
    }


def inspect_all_scenarios(data_dir: str = "data") -> Dict[str, ScenarioInfo]:
    """Inspects all scenario subdirectories in data_dir."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    results: Dict[str, ScenarioInfo] = {}
    for entry in sorted(os.listdir(data_dir)):
        subpath = os.path.join(data_dir, entry)
        if os.path.isdir(subpath):
            try:
                results[entry] = inspect_scenario(entry, data_dir)
            except Exception:
                pass
    return results
