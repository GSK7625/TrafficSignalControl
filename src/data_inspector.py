"""Module for inspecting SUMO network and traffic signal configuration."""

from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional
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

    def __post_init__(self) -> None:
        if self.duration is None:
            self.duration = float(self.end_time - self.begin_time)
        if self.tls_count is None:
            self.tls_count = len(self.tls_ids)
        if self.rou_path is None:
            self.rou_path = self.rou_files[0] if self.rou_files else ""
        elif not self.rou_files and self.rou_path:
            self.rou_files = [self.rou_path]

    def get_green_phases(self, tls_id: Optional[str] = None) -> List[PhaseInfo]:
        """Returns all green phases for a given TLS ID (or the first TLS if omitted)."""
        target_id = tls_id or (self.tls_ids[0] if self.tls_ids else "")
        if not target_id or target_id not in self.phases_per_tls:
            return []
        return [p for p in self.phases_per_tls[target_id] if p.is_green]

    def get_yellow_phases(self, tls_id: Optional[str] = None) -> List[PhaseInfo]:
        """Returns all yellow phases for a given TLS ID (or the first TLS if omitted)."""
        target_id = tls_id or (self.tls_ids[0] if self.tls_ids else "")
        if not target_id or target_id not in self.phases_per_tls:
            return []
        return [p for p in self.phases_per_tls[target_id] if p.is_yellow]


def _parse_departures(rou_file: str) -> List[float]:
    """Parses departure times of trips and vehicles in a route file."""
    root = ET.parse(rou_file).getroot()
    departures = [
        float(item.attrib["depart"])
        for item in root
        if item.tag in ("trip", "vehicle") and "depart" in item.attrib
    ]
    if any(not math.isfinite(val) or val < 0 for val in departures):
        raise ValueError("Departure times must be finite and non-negative")
    return departures


def inspect_scenario(
    scenario_name: str,
    data_dir: str = "data",
    cfg_path: Optional[str] = None,
) -> ScenarioInfo:
    """Inspects a scenario directory or sumocfg file and extracts network metadata."""
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
    tl_nodes = net_root.findall("tlLogic")
    tls_ids = list(dict.fromkeys(tl.get("id") for tl in tl_nodes if tl.get("id")))

    phases_per_tls: Dict[str, List[PhaseInfo]] = {}
    for tl in tl_nodes:
        tl_id = tl.get("id")
        if tl_id and tl_id not in phases_per_tls:
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

    controlled_lanes_per_tls: Dict[str, List[str]] = {}
    controlled_links_per_tls: Dict[str, int] = {}
    for tl_id in tls_ids:
        connections = net_root.findall(f".//connection[@tl='{tl_id}']")
        phases = phases_per_tls.get(tl_id, [])
        controlled_links_per_tls[tl_id] = len(phases[0].state) if phases and phases[0].state else len(connections)
        controlled_lanes_per_tls[tl_id] = sorted({
            f"{c.get('from')}_{c.get('fromLane')}"
            for c in connections
            if c.get("from") and c.get("fromLane") is not None
        })

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
        controlled_lanes_per_tls=controlled_lanes_per_tls,
        controlled_links_per_tls=controlled_links_per_tls,
        trips_or_vehicles_count=demand_count,
        rou_files=rou_files,
        departure_times=all_departs,
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
