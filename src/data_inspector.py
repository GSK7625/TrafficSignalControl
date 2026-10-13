"""Module for inspecting and validating RESCO SUMO dataset files."""

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
    """Comprehensive information regarding a SUMO scenario."""
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


def inspect_scenario(scenario_name: str, data_dir: str = "data") -> ScenarioInfo:
    """Inspects a single scenario directory and extracts all configuration, network,

    and traffic demand metadata.

    Args:
        scenario_name: Name of the scenario folder under data_dir.
        data_dir: Path to root data directory.

    Returns:
        ScenarioInfo object populated with all extracted details.

    Raises:
        FileNotFoundError: If scenario folder or essential files are missing.
        ValueError: If files are malformed or missing key elements.
    """
    scenario_path = os.path.join(data_dir, scenario_name)
    if not os.path.isdir(scenario_path):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_path}")

    cfg_file = os.path.join(scenario_path, f"{scenario_name}.sumocfg")
    if not os.path.isfile(cfg_file):
        # Look for any .sumocfg in scenario_path
        candidates = [f for f in os.listdir(scenario_path) if f.endswith(".sumocfg")]
        if candidates:
            cfg_file = os.path.join(scenario_path, candidates[0])
        else:
            raise FileNotFoundError(f"No .sumocfg found in {scenario_path}")

    # Parse .sumocfg
    cfg_tree = ET.parse(cfg_file)
    cfg_root = cfg_tree.getroot()

    net_node = cfg_root.find(".//net-file")
    if net_node is None or not net_node.get("value"):
        raise ValueError(f"No net-file specified in {cfg_file}")
    net_rel = net_node.get("value")
    net_file = os.path.normpath(os.path.join(scenario_path, net_rel))
    if not os.path.isfile(net_file):
        raise FileNotFoundError(f"Referenced net-file not found: {net_file}")

    # Parse route-files (handles single or comma-separated list)
    route_node = cfg_root.find(".//route-files")
    rou_files: List[str] = []
    if route_node is not None and route_node.get("value"):
        for part in route_node.get("value").split(","):
            part = part.strip()
            if part:
                p = os.path.normpath(os.path.join(scenario_path, part))
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Referenced route-file not found: {p}")
                rou_files.append(p)

    rou_file = rou_files[0] if rou_files else ""

    begin_node = cfg_root.find(".//begin")
    begin_time = float(begin_node.get("value", 0.0)) if begin_node is not None else 0.0

    end_node = cfg_root.find(".//end")
    end_time = float(end_node.get("value", 3600.0)) if end_node is not None else 3600.0
    duration = end_time - begin_time

    # Parse .net.xml
    net_tree = ET.parse(net_file)
    net_root = net_tree.getroot()

    tl_nodes = net_root.findall("tlLogic")
    tls_ids = [tl.get("id") for tl in tl_nodes if tl.get("id")]
    phases_per_tls: Dict[str, List[PhaseInfo]] = {}

    for tl in tl_nodes:
        tl_id = tl.get("id")
        phases = []
        for idx, p in enumerate(tl.findall("phase")):
            dur = float(p.get("duration", 0.0))
            st = p.get("state", "")
            min_d = float(p.get("minDur")) if p.get("minDur") is not None else None
            max_d = float(p.get("maxDur")) if p.get("maxDur") is not None else None
            phases.append(PhaseInfo(index=idx, duration=dur, state=st, min_dur=min_d, max_dur=max_d))
        phases_per_tls[tl_id] = phases

    controlled_lanes_per_tls: Dict[str, List[str]] = {}
    controlled_links_per_tls: Dict[str, int] = {}
    for tl_id in tls_ids:
        connections = net_root.findall(f".//connection[@tl='{tl_id}']")
        phases = phases_per_tls.get(tl_id, [])
        if phases and phases[0].state:
            # Signal link count is determined by the length of the signal state string
            controlled_links_per_tls[tl_id] = len(phases[0].state)
        else:
            link_indices = set(c.get("linkIndex") for c in connections if c.get("linkIndex") is not None)
            controlled_links_per_tls[tl_id] = len(link_indices) if link_indices else len(connections)

        lanes = sorted(list(set(
            f"{c.get('from')}_{c.get('fromLane')}"
            for c in connections
            if c.get("from") and c.get("fromLane") is not None
        )))
        controlled_lanes_per_tls[tl_id] = lanes

    # Parse traffic demand across all route files
    demand_count = 0
    depart_range: Optional[Tuple[float, float]] = None
    all_departs: List[float] = []

    for r_file in (rou_files if rou_files else ([rou_file] if rou_file else [])):
        if os.path.isfile(r_file):
            rou_tree = ET.parse(r_file)
            rou_root = rou_tree.getroot()

            trips_and_vehs = rou_root.findall("trip") + rou_root.findall("vehicle")
            demand_count += len(trips_and_vehs)

            flows = rou_root.findall("flow")
            for flow in flows:
                if flow.get("number") is not None:
                    try:
                        demand_count += int(flow.get("number"))
                    except ValueError:
                        demand_count += 1
                elif flow.get("period") is not None:
                    try:
                        b = float(flow.get("begin", 0.0))
                        e = float(flow.get("end", 3600.0))
                        p = float(flow.get("period"))
                        demand_count += max(0, int((e - b) / p))
                    except (ValueError, ZeroDivisionError):
                        demand_count += 1
                elif flow.get("vehsPerHour") is not None:
                    try:
                        b = float(flow.get("begin", 0.0))
                        e = float(flow.get("end", 3600.0))
                        vph = float(flow.get("vehsPerHour"))
                        demand_count += max(0, int((e - b) * (vph / 3600.0)))
                    except ValueError:
                        demand_count += 1
                elif flow.get("probability") is not None:
                    try:
                        b = float(flow.get("begin", 0.0))
                        e = float(flow.get("end", 3600.0))
                        prob = float(flow.get("probability"))
                        demand_count += max(0, int((e - b) * prob))
                    except ValueError:
                        demand_count += 1
                else:
                    demand_count += 1

            for item in trips_and_vehs + flows:
                d = item.get("depart") or item.get("begin")
                if d is not None:
                    try:
                        all_departs.append(float(d))
                    except ValueError:
                        pass

    if all_departs:
        depart_range = (min(all_departs), max(all_departs))

    # Identify auxiliary files
    all_files = os.listdir(scenario_path)
    primary_files = {os.path.basename(cfg_file), os.path.basename(net_file)}
    for rf in rou_files:
        primary_files.add(os.path.basename(rf))
    aux = [f for f in all_files if f not in primary_files]

    return ScenarioInfo(
        name=scenario_name,
        cfg_path=cfg_file,
        net_path=net_file,
        rou_path=rou_file,
        begin_time=begin_time,
        end_time=end_time,
        duration=duration,
        tls_count=len(tls_ids),
        tls_ids=tls_ids,
        phases_per_tls=phases_per_tls,
        controlled_lanes_per_tls=controlled_lanes_per_tls,
        controlled_links_per_tls=controlled_links_per_tls,
        trips_or_vehicles_count=demand_count,
        vehicle_depart_range=depart_range,
        auxiliary_files=aux,
    )


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
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to inspect scenario '{entry}': {e}")
    return results
