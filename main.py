"""Command-line interface for the Traffic Signal Control simulation platform."""

import argparse
import json
import os
import sys

from src.config import get_scenario_config
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.data_inspector import inspect_all_scenarios, inspect_scenario
from src.runner import SimulationRunner


def format_scenario_info(info) -> str:
    """Formats ScenarioInfo into a readable multiline string."""
    lines = [
        f"Scenario: {info.name}",
        f"  Config file: {info.cfg_path}",
        f"  Net file:    {info.net_path}",
        f"  Route file:  {info.rou_path}",
        f"  Time window: [{info.begin_time:.0f}s - {info.end_time:.0f}s] (Duration: {info.duration:.0f}s)",
        f"  Intersections (TLS count): {info.tls_count}",
    ]
    for tl_id in info.tls_ids:
        phases = info.phases_per_tls.get(tl_id, [])
        lanes = info.controlled_lanes_per_tls.get(tl_id, [])
        links = info.controlled_links_per_tls.get(tl_id, 0)
        lines.append(f"    TLS '{tl_id}': {len(phases)} phases, {links} links, {len(lanes)} controlled incoming lanes")
    lines.append(f"  Traffic Demand (Trips/Vehicles): {info.trips_or_vehicles_count}")
    if info.vehicle_depart_range:
        lines.append(f"  Depart Time Range: [{info.vehicle_depart_range[0]:.1f}s - {info.vehicle_depart_range[1]:.1f}s]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="SUMO Traffic Signal Control Platform (RESCO Datasets)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cologne1",
        help="Dataset name (e.g., cologne1, ingolstadt1, cologne3)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable SUMO graphical user interface (sumo-gui)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum simulation steps to run (default: full scenario duration)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="original",
        choices=["original", "cyclic", "agent"],
        help="Signal control mode: 'original' (SUMO native program), 'cyclic' (30s cycle), 'agent' (external control)",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="none",
        choices=["none", "fixed-time", "max-pressure"],
        help="Traffic controller algorithm: 'fixed-time', 'max-pressure', or 'none' (uses --mode)",
    )
    parser.add_argument(
        "--fixed-time-green",
        type=float,
        default=30.0,
        help="Green time duration for Fixed-Time controller in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--pressure-metric",
        type=str,
        default="queue",
        choices=["queue", "vehicle_count"],
        help="Pressure measurement metric for Max-Pressure: 'queue' (halting vehicles) or 'vehicle_count' (all vehicles)",
    )
    parser.add_argument(
        "--yellow-dur",
        type=float,
        default=3.0,
        help="Yellow clearance duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--all-red-dur",
        type=float,
        default=0.0,
        help="All-red clearance duration in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--min-green",
        type=float,
        default=5.0,
        help="Minimum green time constraint in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--step-len",
        type=float,
        default=1.0,
        help="Simulation time step in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--tripinfo",
        action="store_true",
        help="Generate and parse SUMO native --tripinfo-output for ground-truth trip metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save output simulation results (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Simulation random seed (default: 42)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect and print details of the selected dataset without running simulation",
    )
    parser.add_argument(
        "--inspect-all",
        action="store_true",
        help="Inspect and print details for all available datasets in data/",
    )

    args = parser.parse_args()

    if args.inspect_all:
        print("\n================ RESCO DATASETS INSPECTION ================")
        all_info = inspect_all_scenarios(data_dir="data")
        for name, info in all_info.items():
            print(format_scenario_info(info))
            print("-" * 50)
        return

    if args.inspect:
        print("\n================ SCENARIO INSPECTION ================")
        info = inspect_scenario(args.dataset, data_dir="data")
        print(format_scenario_info(info))
        return

    controller = None
    if args.controller == "fixed-time":
        controller = FixedTimeController(default_green_time=args.fixed_time_green)
        effective_mode = "agent"
    elif args.controller == "max-pressure":
        controller = MaxPressureController(min_green=args.min_green, pressure_metric=args.pressure_metric)
        effective_mode = "agent"
    else:
        effective_mode = args.mode

    print(f"\n[INFO] Initializing simulation for dataset: '{args.dataset}'")
    ctrl_desc = f"Controller: '{controller.name}'" if controller else f"Control Mode: '{effective_mode}'"
    print(f"[INFO] Mode: {'GUI (sumo-gui)' if args.gui else 'Headless (sumo)'} | {ctrl_desc}")
    print(
        f"[INFO] Min Green: {args.min_green}s | Yellow: {args.yellow_dur}s | "
        f"All-Red: {args.all_red_dur}s | Step Length: {args.step_len}s"
    )

    cfg = get_scenario_config(
        dataset_name=args.dataset,
        gui=args.gui,
        step_length=args.step_len,
        yellow_duration=args.yellow_dur,
        all_red_duration=args.all_red_dur,
        min_green=args.min_green,
        max_steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        control_mode=effective_mode,
        use_tripinfo=args.tripinfo,
    )

    runner = SimulationRunner(cfg, controller=controller)
    print(f"[INFO] Traffic lights found: {list(runner.tls_managers.keys())}")
    for tls_id, mgr in runner.tls_managers.items():
        print(f"  TLS '{tls_id}': {len(mgr.green_phases)} active green phases configured")

    print("[INFO] Starting simulation...")
    summary = runner.run(num_steps=args.steps)

    print("\n================ SIMULATION RESULTS ================")
    print(json.dumps(summary, indent=2))
    print(f"\n[SUCCESS] Results successfully exported to: '{summary.get('output_json')}'")


if __name__ == "__main__":
    main()
