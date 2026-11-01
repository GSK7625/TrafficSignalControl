"""Module đọc và trích xuất cấu hình mạng lưới và nhu cầu giao thông SUMO.

Được thiết kế đơn giản, trực tiếp và dễ giải thích cho người mới học.
"""

import os
from typing import Any, Dict, List
import xml.etree.ElementTree as ET


def read_scenario(scenario_name: str, data_dir: str = "data") -> Dict[str, Any]:
    """Đọc thông tin mạng lưới, kịch bản và nhu cầu xe từ các file gốc của SUMO.

    Tham số:
        scenario_name: Tên kịch bản (ví dụ 'cologne1', 'ingolstadt1').
        data_dir: Thư mục chứa dữ liệu (mặc định 'data').

    Trả về:
        Dictionary chứa toàn bộ cấu hình cần thiết để mô phỏng và điều khiển.
    """
    scen_dir = os.path.join(data_dir, scenario_name)
    if not os.path.isdir(scen_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục kịch bản: {scen_dir}")

    cfg_file = os.path.join(scen_dir, f"{scenario_name}.sumocfg")
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {cfg_file}")

    # 1. Đọc file cấu hình .sumocfg để lấy thời gian mô phỏng và tên file mạng/route
    cfg_tree = ET.parse(cfg_file)
    cfg_root = cfg_tree.getroot()

    net_node = cfg_root.find(".//net-file")
    rou_node = cfg_root.find(".//route-files")
    begin_node = cfg_root.find(".//begin")
    end_node = cfg_root.find(".//end")

    net_filename = net_node.get("value") if net_node is not None else f"{scenario_name}.net.xml"
    rou_filename = rou_node.get("value") if rou_node is not None else f"{scenario_name}.rou.xml"

    begin_time = float(begin_node.get("value", 0.0)) if begin_node is not None else 0.0
    end_time = float(end_node.get("value", 3600.0)) if end_node is not None else 3600.0

    net_file = os.path.join(scen_dir, net_filename)
    rou_file = os.path.join(scen_dir, rou_filename)

    # 2. Đọc file mạng .net.xml để lấy thông tin đèn tín hiệu và các làn đường
    net_tree = ET.parse(net_file)
    net_root = net_tree.getroot()

    tl_nodes = net_root.findall("tlLogic")
    if not tl_nodes:
        raise ValueError(f"Không tìm thấy bộ điều khiển đèn (tlLogic) trong {net_file}")

    # Lấy bộ điều khiển đèn đầu tiên (cả 2 map đều là đơn nút giao)
    tl_node = tl_nodes[0]
    tls_id = tl_node.get("id", "")

    # Trích xuất danh sách tất cả các pha gốc trong XML
    raw_phases = [p.get("state", "") for p in tl_node.findall("phase")]

    # Lọc pha xanh hợp lệ: không chứa đèn vàng 'y', có chứa tín hiệu cho phép đi ('G', 'g', 's')
    green_phases = [p for p in raw_phases if "y" not in p and any(c in "Ggs" for c in p)]

    # Trích xuất các kết nối (connections) được điều khiển bởi đèn
    conns = [c for c in net_root.findall(".//connection") if c.get("tl") == tls_id]
    conns.sort(key=lambda c: int(c.get("linkIndex", -1)))

    incoming_lanes_set = set()
    outgoing_lanes_set = set()
    movements: List[Dict[str, Any]] = []

    for c in conns:
        link_idx = int(c.get("linkIndex", -1))
        from_edge = c.get("from", "")
        to_edge = c.get("to", "")
        from_lane = f"{from_edge}_{c.get('fromLane')}"
        to_lane = f"{to_edge}_{c.get('toLane')}"
        direction = c.get("dir", "")

        incoming_lanes_set.add(from_lane)
        outgoing_lanes_set.add(to_lane)
        movements.append({
            "link_index": link_idx,
            "from_lane": from_lane,
            "to_lane": to_lane,
            "dir": direction,
        })

    # 3. Đọc file luồng xe .rou.xml để đếm tổng nhu cầu
    rou_tree = ET.parse(rou_file)
    rou_root = rou_tree.getroot()
    all_trips = rou_root.findall("trip") + rou_root.findall("vehicle")
    total_demand = len(all_trips)

    return {
        "name": scenario_name,
        "cfg_path": cfg_file,
        "net_path": net_file,
        "rou_path": rou_file,
        "begin_time": begin_time,
        "end_time": end_time,
        "duration": end_time - begin_time,
        "tls_id": tls_id,
        "green_phases": green_phases,
        "all_raw_phases": raw_phases,
        "incoming_lanes": sorted(list(incoming_lanes_set)),
        "outgoing_lanes": sorted(list(outgoing_lanes_set)),
        "movements": movements,
        "total_demand": total_demand,
    }
