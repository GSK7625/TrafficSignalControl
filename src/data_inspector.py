"""Module phân tích và trích xuất cấu hình mạng lưới giao thông và đèn tín hiệu SUMO."""

from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


@dataclass
class PhaseInfo:
    """Thông tin về một pha tín hiệu đèn giao thông."""
    index: int
    duration: float
    state: str
    min_dur: Optional[float] = None
    max_dur: Optional[float] = None

    @property
    def is_yellow(self) -> bool:
        """Trả về True nếu pha có chứa bất kỳ tín hiệu đèn vàng nào."""
        return "y" in self.state.lower()

    @property
    def is_green(self) -> bool:
        """Trả về True nếu pha chứa tín hiệu đèn xanh và không có đèn vàng."""
        return not self.is_yellow and any(c in "Gg" for c in self.state)


@dataclass
class LaneInfo:
    """Thuộc tính vật lý của một làn đường trong mạng lưới."""
    id: str
    speed: float  # Tốc độ tối đa cho phép tính bằng m/s
    length: float  # Chiều dài làn đường tính bằng mét
    edge_id: str = ""


@dataclass
class MovementInfo:
    """Thông tin về luồng chuyển động / liên kết được điều khiển tại nút giao."""
    link_index: int
    from_lane: str
    to_lane: str
    direction: str  # 's' (đi thẳng), 'l' (rẽ trái), 'r' (rẽ phải), 't' (quay đầu), v.v.
    from_edge: str = ""
    to_edge: str = ""
    state: str = ""


@dataclass
class ScenarioInfo:
    """Thông tin toàn diện về một kịch bản mô phỏng SUMO."""
    name: str
    cfg_path: str
    net_path: str
    rou_path: Optional[str] = None
    begin_time: float = 0.0
    end_time: float = 3600.0
    duration: Optional[float] = None
    tls_count: Optional[int] = None
    tls_ids: List[str] = field(default_factory=list)
    phases_per_tls: Dict[str, List[PhaseInfo]] = field(default_factory=dict)
    controlled_lanes_per_tls: Dict[str, List[str]] = field(default_factory=dict)
    controlled_links_per_tls: Dict[str, int] = field(default_factory=dict)
    trips_or_vehicles_count: int = 0
    rou_files: List[str] = field(default_factory=list)
    departure_times: Optional[List[float]] = None

    # Các thuộc tính mạng lưới mở rộng cho Max-Pressure, RL và tọa độ không gian
    incoming_lanes_per_tls: Dict[str, List[str]] = field(default_factory=dict)
    outgoing_lanes_per_tls: Dict[str, List[str]] = field(default_factory=dict)
    movements_per_tls: Dict[str, List[MovementInfo]] = field(default_factory=dict)
    lanes: Dict[str, LaneInfo] = field(default_factory=dict)
    junction_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    tls_types: Dict[str, str] = field(default_factory=dict)
    tls_programs: Dict[str, str] = field(default_factory=dict)
    boundary: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        if self.duration is None:
            self.duration = float(self.end_time - self.begin_time)
        if self.tls_count is None:
            self.tls_count = len(self.tls_ids)
        if self.rou_path is None:
            self.rou_path = self.rou_files[0] if self.rou_files else ""
        elif not self.rou_files and self.rou_path:
            self.rou_files = [self.rou_path]

        # Đồng bộ controlled_lanes_per_tls với incoming_lanes_per_tls để tương thích ngược
        if not self.controlled_lanes_per_tls and self.incoming_lanes_per_tls:
            self.controlled_lanes_per_tls = self.incoming_lanes_per_tls
        elif not self.incoming_lanes_per_tls and self.controlled_lanes_per_tls:
            self.incoming_lanes_per_tls = self.controlled_lanes_per_tls

    def _resolve_tls_id(self, tls_id: Optional[str] = None) -> str:
        """Xác định target TLS ID, mặc định chọn TLS đầu tiên nếu bỏ qua."""
        return tls_id or (self.tls_ids[0] if self.tls_ids else "")

    def get_green_phases(self, tls_id: Optional[str] = None) -> List[PhaseInfo]:
        """Trả về tất cả các pha đèn xanh của TLS ID đã cho (hoặc TLS đầu tiên nếu bỏ qua)."""
        target_id = self._resolve_tls_id(tls_id)
        if not target_id or target_id not in self.phases_per_tls:
            return []
        return [p for p in self.phases_per_tls[target_id] if p.is_green]

    def get_yellow_phases(self, tls_id: Optional[str] = None) -> List[PhaseInfo]:
        """Trả về tất cả các pha đèn vàng của TLS ID đã cho (hoặc TLS đầu tiên nếu bỏ qua)."""
        target_id = self._resolve_tls_id(tls_id)
        if not target_id or target_id not in self.phases_per_tls:
            return []
        return [p for p in self.phases_per_tls[target_id] if p.is_yellow]

    def get_incoming_lanes(self, tls_id: Optional[str] = None) -> List[str]:
        """Trả về danh sách các làn đi vào của TLS ID đã cho (hoặc TLS đầu tiên nếu bỏ qua)."""
        target_id = self._resolve_tls_id(tls_id)
        return self.incoming_lanes_per_tls.get(target_id, [])

    def get_outgoing_lanes(self, tls_id: Optional[str] = None) -> List[str]:
        """Trả về danh sách các làn thoát ra của TLS ID đã cho (hoặc TLS đầu tiên nếu bỏ qua)."""
        target_id = self._resolve_tls_id(tls_id)
        return self.outgoing_lanes_per_tls.get(target_id, [])

    def get_movements(self, tls_id: Optional[str] = None) -> List[MovementInfo]:
        """Trả về chi tiết các luồng chuyển động của TLS ID đã cho (hoặc TLS đầu tiên nếu bỏ qua)."""
        target_id = self._resolve_tls_id(tls_id)
        return self.movements_per_tls.get(target_id, [])

    def get_lane_length(self, lane_id: str) -> float:
        """Trả về chiều dài làn đường tính bằng mét (0.0 nếu không tìm thấy)."""
        lane = self.lanes.get(lane_id)
        return lane.length if lane else 0.0

    def get_lane_speed(self, lane_id: str) -> float:
        """Trả về tốc độ tối đa cho phép trên làn tính bằng m/s (0.0 nếu không tìm thấy)."""
        lane = self.lanes.get(lane_id)
        return lane.speed if lane else 0.0


def _parse_departures(rou_file: str) -> List[float]:
    """Phân tích thời điểm xuất phát của trips, vehicles, và flows trong file lộ trình routes."""
    root = ET.parse(rou_file).getroot()
    departures: List[float] = []

    for item in root:
        if item.tag in ("trip", "vehicle") and "depart" in item.attrib:
            departures.append(float(item.attrib["depart"]))
        elif item.tag == "flow":
            begin = float(item.attrib.get("begin", 0.0))
            end = float(item.attrib.get("end", 0.0))
            if "number" in item.attrib:
                num = int(item.attrib["number"])
                if num > 0:
                    step = (end - begin) / num if end > begin else 0.0
                    departures.extend(begin + i * step for i in range(num))
            elif "period" in item.attrib or "vehsPerHour" in item.attrib:
                period = (
                    float(item.attrib["period"])
                    if "period" in item.attrib
                    else (3600.0 / float(item.attrib["vehsPerHour"]) if float(item.attrib["vehsPerHour"]) > 0 else 0.0)
                )
                if period > 0:
                    t = begin
                    while t < end:
                        departures.append(t)
                        t += period
            elif "depart" in item.attrib:
                departures.append(float(item.attrib["depart"]))
            elif "begin" in item.attrib:
                departures.append(begin)

    departures.sort()
    if any(not math.isfinite(val) or val < 0 for val in departures):
        raise ValueError("Departure times must be finite and non-negative")
    return departures


def inspect_scenario(
    scenario_name: str,
    data_dir: str = "data",
    cfg_path: Optional[str] = None,
) -> ScenarioInfo:
    """Phân tích thư mục kịch bản hoặc file .sumocfg và trích xuất đầy đủ metadata mạng lưới."""
    if cfg_path:
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"SUMO configuration not found: {cfg_path}")
        cfg_file = os.path.abspath(cfg_path)
        scenario_path = os.path.dirname(cfg_file)
    else:
        scenario_path = os.path.join(data_dir, scenario_name)
        if not os.path.isdir(scenario_path):
            raise FileNotFoundError(f"Scenario directory not found: {scenario_path}")

        cfg_file = os.path.join(scenario_path, f"{scenario_name}.sumocfg")
        if not os.path.isfile(cfg_file):
            candidates = [f for f in os.listdir(scenario_path) if f.endswith(".sumocfg")]
            if candidates:
                cfg_file = os.path.join(scenario_path, candidates[0])
            else:
                raise FileNotFoundError(f"No .sumocfg file found in: {scenario_path}")

    cfg_root = ET.parse(cfg_file).getroot()
    net_node = cfg_root.find(".//net-file")
    if net_node is None or not net_node.get("value"):
        raise ValueError(f"Missing required <net-file> entry in configuration: {cfg_file}")
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

    # 1. Ranh giới mạng lưới (Boundary)
    boundary: Optional[Tuple[float, float, float, float]] = None
    loc_node = net_root.find("location")
    if loc_node is not None and "convBoundary" in loc_node.attrib:
        try:
            parts = [float(x) for x in loc_node.attrib["convBoundary"].split(",")]
            if len(parts) == 4:
                boundary = (parts[0], parts[1], parts[2], parts[3])
        except (ValueError, IndexError):
            boundary = None

    # 2. Thuộc tính vật lý của các làn đường (chiều dài, giới hạn tốc độ, edge_id)
    lanes: Dict[str, LaneInfo] = {}
    for edge in net_root.findall("edge"):
        edge_id = edge.get("id", "")
        # Bỏ qua các đoạn nối nội bộ bắt đầu bằng dấu ':'
        if edge_id.startswith(":"):
            continue
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            if lane_id:
                lanes[lane_id] = LaneInfo(
                    id=lane_id,
                    speed=float(lane.get("speed", 13.89)),
                    length=float(lane.get("length", 0.0)),
                    edge_id=edge_id,
                )

    # 3. Tọa độ các nút giao (Junction Positions)
    junction_positions: Dict[str, Tuple[float, float]] = {}
    for junc in net_root.findall("junction"):
        jid = junc.get("id")
        if jid and "x" in junc.attrib and "y" in junc.attrib:
            try:
                junction_positions[jid] = (float(junc.attrib["x"]), float(junc.attrib["y"]))
            except ValueError:
                pass

    # 4. Hệ thống đèn tín hiệu giao thông (TLS)
    tl_nodes = net_root.findall("tlLogic")
    tls_ids = list(dict.fromkeys(tl.get("id") for tl in tl_nodes if tl.get("id")))

    phases_per_tls: Dict[str, List[PhaseInfo]] = {}
    tls_types: Dict[str, str] = {}
    tls_programs: Dict[str, str] = {}
    for tl in tl_nodes:
        tl_id = tl.get("id")
        if tl_id and tl_id not in phases_per_tls:
            tls_types[tl_id] = tl.get("type", "static")
            tls_programs[tl_id] = tl.get("programID", "0")
            phases_per_tls[tl_id] = [
                PhaseInfo(
                    index=idx,
                    duration=float(p.get("duration", 0)),
                    state=p.get("state", ""),
                    min_dur=float(p.get("minDur")) if p.get("minDur") is not None else None,
                    max_dur=float(p.get("maxDur")) if p.get("maxDur") is not None else None,
                )
                for idx, p in enumerate(tl.findall("phase"))
            ]

    # 5. Liên kết được điều khiển, làn vào & làn ra, và ánh xạ luồng chuyển động (Movements)
    incoming_lanes_per_tls: Dict[str, List[str]] = {}
    outgoing_lanes_per_tls: Dict[str, List[str]] = {}
    movements_per_tls: Dict[str, List[MovementInfo]] = {}
    controlled_links_per_tls: Dict[str, int] = {}

    all_connections = net_root.findall(".//connection")
    for tl_id in tls_ids:
        tl_conns = [c for c in all_connections if c.get("tl") == tl_id]
        phases = phases_per_tls.get(tl_id, [])
        controlled_links_per_tls[tl_id] = (
            len(phases[0].state) if phases and phases[0].state else len(tl_conns)
        )

        in_lanes = set()
        out_lanes = set()
        movements: List[MovementInfo] = []

        for c in tl_conns:
            from_edge = c.get("from", "")
            to_edge = c.get("to", "")
            from_lane_idx = c.get("fromLane")
            to_lane_idx = c.get("toLane")

            from_lane = f"{from_edge}_{from_lane_idx}" if from_edge and from_lane_idx is not None else ""
            to_lane = f"{to_edge}_{to_lane_idx}" if to_edge and to_lane_idx is not None else ""

            if from_lane:
                in_lanes.add(from_lane)
            if to_lane:
                out_lanes.add(to_lane)

            link_idx_str = c.get("linkIndex")
            link_index = int(link_idx_str) if link_idx_str is not None else -1

            movements.append(
                MovementInfo(
                    link_index=link_index,
                    from_lane=from_lane,
                    to_lane=to_lane,
                    direction=c.get("dir", ""),
                    from_edge=from_edge,
                    to_edge=to_edge,
                    state=c.get("state", ""),
                )
            )

        # Sắp xếp movements theo link_index để đảm bảo thứ tự vị trí nhất quán
        movements.sort(key=lambda m: m.link_index)
        movements_per_tls[tl_id] = movements
        incoming_lanes_per_tls[tl_id] = sorted(in_lanes)
        outgoing_lanes_per_tls[tl_id] = sorted(out_lanes)

    # 6. Phân tích nhu cầu lưu lượng phương tiện (Demand)
    all_departs: List[float] = []
    for r_file in rou_files:
        if os.path.exists(r_file):
            all_departs.extend(_parse_departures(r_file))
    demand_count = len(all_departs)

    return ScenarioInfo(
        name=scenario_name,
        cfg_path=cfg_file,
        net_path=net_file,
        rou_path=rou_file,
        begin_time=begin_time,
        end_time=end_time,
        tls_ids=tls_ids,
        phases_per_tls=phases_per_tls,
        controlled_lanes_per_tls=incoming_lanes_per_tls,
        controlled_links_per_tls=controlled_links_per_tls,
        trips_or_vehicles_count=demand_count,
        rou_files=rou_files,
        departure_times=all_departs,
        incoming_lanes_per_tls=incoming_lanes_per_tls,
        outgoing_lanes_per_tls=outgoing_lanes_per_tls,
        movements_per_tls=movements_per_tls,
        lanes=lanes,
        junction_positions=junction_positions,
        tls_types=tls_types,
        tls_programs=tls_programs,
        boundary=boundary,
    )


