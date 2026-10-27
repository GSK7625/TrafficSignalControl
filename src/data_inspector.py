"""Module for inspecting SUMO network and traffic signal configuration."""

from dataclasses import dataclass, field
import math
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

    @property
    def is_yellow(self) -> bool:
        """Returns True if the phase contains any yellow signal."""
        return "y" in self.state.lower()

    @property
    def is_green(self) -> bool:
        """Returns True if the phase contains green signal and no yellow."""
        return not self.is_yellow and any(c in "Gg" for c in self.state)


@dataclass
class LaneInfo:
    """Physical properties of a network lane."""
    id: str
    speed: float  # Maximum speed in m/s
    length: float  # Lane length in meters
    edge_id: str = ""


@dataclass
class MovementInfo:
    """Information regarding a controlled link/movement at a traffic light."""
    link_index: int
    from_lane: str
    to_lane: str
    direction: str  # 's' (straight), 'l' (left), 'r' (right), 't' (turn), etc.
    from_edge: str = ""
    to_edge: str = ""
    state: str = ""


@dataclass
class ScenarioInfo:
    """Information regarding a SUMO scenario."""
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

    # Extended network attributes for Max-Pressure, RL, and spatial coordination
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

        # Sync controlled_lanes_per_tls with incoming_lanes_per_tls for backward compatibility
        if not self.controlled_lanes_per_tls and self.incoming_lanes_per_tls:
            self.controlled_lanes_per_tls = self.incoming_lanes_per_tls
        elif not self.incoming_lanes_per_tls and self.controlled_lanes_per_tls:
            self.incoming_lanes_per_tls = self.controlled_lanes_per_tls

    def _resolve_tls_id(self, tls_id: Optional[str] = None) -> str:
        """Resolves target TLS ID, defaulting to the first TLS if omitted."""
        return tls_id or (self.tls_ids[0] if self.tls_ids else "")

    def get_green_phases(self, tls_id: Optional[str] = None) -> List[PhaseInfo]:
        """Returns all green phases for a given TLS ID (or the first TLS if omitted)."""
        target_id = self._resolve_tls_id(tls_id)
        if not target_id or target_id not in self.phases_per_tls:
            return []
        return [p for p in self.phases_per_tls[target_id] if p.is_green]

    def get_yellow_phases(self, tls_id: Optional[str] = None) -> List[PhaseInfo]:
        """Returns all yellow phases for a given TLS ID (or the first TLS if omitted)."""
        target_id = self._resolve_tls_id(tls_id)
        if not target_id or target_id not in self.phases_per_tls:
            return []
        return [p for p in self.phases_per_tls[target_id] if p.is_yellow]

    def get_incoming_lanes(self, tls_id: Optional[str] = None) -> List[str]:
        """Returns incoming lanes for a given TLS ID (or the first TLS if omitted)."""
        target_id = self._resolve_tls_id(tls_id)
        return self.incoming_lanes_per_tls.get(target_id, [])

    def get_outgoing_lanes(self, tls_id: Optional[str] = None) -> List[str]:
        """Returns outgoing (exit) lanes for a given TLS ID (or the first TLS if omitted)."""
        target_id = self._resolve_tls_id(tls_id)
        return self.outgoing_lanes_per_tls.get(target_id, [])

    def get_movements(self, tls_id: Optional[str] = None) -> List[MovementInfo]:
        """Returns movement/link details for a given TLS ID (or the first TLS if omitted)."""
        target_id = self._resolve_tls_id(tls_id)
        return self.movements_per_tls.get(target_id, [])

    def get_lane_length(self, lane_id: str) -> float:
        """Returns the length of a lane in meters (0.0 if not found)."""
        lane = self.lanes.get(lane_id)
        return lane.length if lane else 0.0

    def get_lane_speed(self, lane_id: str) -> float:
        """Returns maximum speed limit of a lane in m/s (0.0 if not found)."""
        lane = self.lanes.get(lane_id)
        return lane.speed if lane else 0.0


def _parse_departures(rou_file: str) -> List[float]:
    """Parses departure times of trips, vehicles, and flows in a route file."""
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
            elif "period" in item.attrib:
                period = float(item.attrib["period"])
                if period > 0:
                    t = begin
                    while t < end:
                        departures.append(t)
                        t += period
            elif "vehsPerHour" in item.attrib:
                vph = float(item.attrib["vehsPerHour"])
                if vph > 0:
                    period = 3600.0 / vph
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
    """Inspects a scenario directory or sumocfg file and extracts full network metadata."""
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

    # 1. Network Boundary
    boundary: Optional[Tuple[float, float, float, float]] = None
    loc_node = net_root.find("location")
    if loc_node is not None and "convBoundary" in loc_node.attrib:
        try:
            parts = [float(x) for x in loc_node.attrib["convBoundary"].split(",")]
            if len(parts) == 4:
                boundary = (parts[0], parts[1], parts[2], parts[3])
        except (ValueError, IndexError):
            boundary = None

    # 2. Lanes Physical Attributes (length, speed limit, edge_id)
    lanes: Dict[str, LaneInfo] = {}
    for edge in net_root.findall("edge"):
        edge_id = edge.get("id", "")
        # Skip internal edges starting with ':'
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

    # 3. Junction Positions
    junction_positions: Dict[str, Tuple[float, float]] = {}
    for junc in net_root.findall("junction"):
        jid = junc.get("id")
        if jid and "x" in junc.attrib and "y" in junc.attrib:
            try:
                junction_positions[jid] = (float(junc.attrib["x"]), float(junc.attrib["y"]))
            except ValueError:
                pass

    # 4. Traffic Light Systems (TLS)
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

    # 5. Controlled Links, Inflow & Outflow Lanes, and Movement Mapping
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

        # Sort movements by link_index for reliable positional alignment
        movements.sort(key=lambda m: m.link_index)
        movements_per_tls[tl_id] = movements
        incoming_lanes_per_tls[tl_id] = sorted(in_lanes)
        outgoing_lanes_per_tls[tl_id] = sorted(out_lanes)

    # 6. Demand Parsing
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


def get_window_demand(
    scenario_name_or_info: Any,
    window_begin: Optional[float] = None,
    window_end: Optional[float] = None,
    data_dir: str = "data",
) -> Dict[str, Any]:
    """Calculates scheduled traffic demand within [window_begin, window_end)."""
    info = (
        scenario_name_or_info
        if isinstance(scenario_name_or_info, ScenarioInfo)
        else inspect_scenario(scenario_name_or_info, data_dir)
    )
    w_begin = info.begin_time if window_begin is None else float(window_begin)
    w_end = info.end_time if window_end is None else float(window_end)

    departures = info.departure_times
    if departures is None:
        departures = []
        for path in info.rou_files or ([info.rou_path] if info.rou_path else []):
            if os.path.exists(path):
                departures.extend(_parse_departures(path))

    count = sum(w_begin <= departure < w_end for departure in departures)
    return {
        "scheduled_demand_in_window": count,
        "demand_counting_mode": "exact",
        "window_begin": float(w_begin),
        "window_end": float(w_end),
        "window_duration": max(0.0, float(w_end) - float(w_begin)),
        "total_route_demand": info.trips_or_vehicles_count,
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
                continue
    return results
