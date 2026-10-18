"""Command-line interface for the Traffic Signal Control simulation platform."""

import argparse
import json
import os
import sys

from src.config import get_scenario_config
from src.controllers import FixedTimeController, MaxPressureController, QLearningController
from src.data_inspector import inspect_all_scenarios, inspect_scenario
from src.runner import SimulationRunner, extract_green_phases


def format_scenario_info(info) -> str:
    """Formats ScenarioInfo into a readable multiline string."""
    lines = [
        f"Scenario: {info.name}",
        f"  Config file: {info.cfg_path}",
        f"  Net file:    {info.net_path}",
        f"  Route file:  {info.rou_path}",
        f"  Time window: [{info.begin_time:.0f}s - {info.end_time:.0f}s] (Duration: {info.duration:.0f}s)",
        f"  Intersections: {info.tls_count} (IDs: {info.tls_ids})",
        f"  Traffic Demand: {info.trips_or_vehicles_count} vehicles",
    ]
    for tl_id in info.tls_ids:
        phases = info.phases_per_tls.get(tl_id, [])
        lanes = info.controlled_lanes_per_tls.get(tl_id, [])
        lines.append(f"    TLS '{tl_id}': {len(phases)} phases, {len(lanes)} incoming lanes")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="SUMO Traffic Signal Control Platform")
    parser.add_argument("--dataset", default="cologne1", help="Dataset name (e.g. cologne1, ingolstadt1)")
    parser.add_argument("--algorithm", "--controller", dest="controller", default="none",
                        choices=["none", "fixed-time", "max-pressure", "q-learning"], help="Traffic controller algorithm")
    parser.add_argument("--mode", default="original", choices=["original", "cyclic", "agent"],
                        help="Signal control mode: original, cyclic, agent")
    parser.add_argument("--steps", type=int, default=None, help="Max simulation steps")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO-GUI")
    parser.add_argument("--gui-delay", type=float, default=0.0, help="SUMO-GUI delay (ms)")
    parser.add_argument("--model", default=None, help="Path to trained Q-Learning model (.json)")
    parser.add_argument("--fixed-time-green", type=float, default=30.0, help="Green duration for Fixed-Time (s)")
    parser.add_argument("--pressure-metric", default="queue", choices=["queue", "vehicle_count"],
                        help="Pressure metric for Max-Pressure")
    parser.add_argument("--min-green", type=float, default=None, help="Min green time (s)")
    parser.add_argument("--max-green", type=float, default=None, help="Max green time (s)")
    parser.add_argument("--yellow-dur", type=float, default=None, help="Yellow clearance (s)")
    parser.add_argument("--all-red-dur", type=float, default=None, help="All-red clearance (s)")
    parser.add_argument("--step-len", type=float, default=1.0, help="Simulation step length (s)")
    parser.add_argument("--decision-interval", type=float, default=None, help="Decision interval (s)")
    parser.add_argument("--tripinfo", action="store_true", help="Generate tripinfo output")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-tls", default=None, help="Target TLS ID")
    parser.add_argument("--no-libsumo", dest="use_libsumo", action="store_false", default=True,
                        help="Force TraCI socket over libsumo")
    parser.add_argument("--inspect", action="store_true", help="Inspect selected dataset")
    parser.add_argument("--inspect-all", action="store_true", help="Inspect all datasets")
    parser.add_argument("--allow-untrained", action="store_true", help="Allow untrained Q-learning")
    parser.add_argument("--save-details", action="store_true", help="Save detailed simulation files (XML/CSV)")
    return parser.parse_args()


def build_controller(args, info):
    """Instantiates controller based on CLI args."""
    if args.controller == "fixed-time":
        return FixedTimeController(default_green_time=args.fixed_time_green), "agent"

    if args.controller == "max-pressure":
        return MaxPressureController(
            min_green=args.min_green or 5.0,
            max_green=args.max_green or 60.0,
            pressure_metric=args.pressure_metric,
        ), "agent"

    if args.controller == "q-learning":
        tl_id = args.target_tls or info.tls_ids[0]
        green_phases = extract_green_phases(args.dataset, tl_id, info.phases_per_tls.get(tl_id, []))
        ctrl = QLearningController(num_actions=len(green_phases), target_tls_id=tl_id, seed=args.seed, eval_mode=True)
        model_path = args.model or f"models/q_learning/q_learning_{args.dataset}_best.json"
        if os.path.isfile(model_path):
            ctrl.load(model_path)
            print(f"[INFO] Loaded Q-Learning model from '{model_path}'")
        ctrl.eval_mode = True
        ctrl.epsilon = 0.0
        return ctrl, "agent"

    return None, args.mode


def main():
    args = parse_args()

    if args.inspect_all:
        for name, info in inspect_all_scenarios(data_dir="data").items():
            print(format_scenario_info(info))
            print("-" * 50)
        return

    if args.inspect:
        info = inspect_scenario(args.dataset, data_dir="data")
        print(format_scenario_info(info))
        return

    info = inspect_scenario(args.dataset, data_dir="data")
    controller, mode = build_controller(args, info)

    min_green = args.min_green or getattr(controller, "min_green", None) or 5.0
    max_green = args.max_green or getattr(controller, "max_green", None) or 60.0
    yellow_dur = args.yellow_dur or getattr(controller, "yellow_duration", None) or 3.0
    all_red_dur = args.all_red_dur or getattr(controller, "all_red_duration", None) or 1.0
    dec_interval = (
        args.decision_interval
        or getattr(controller, "decision_interval", None)
        or (5.0 if args.controller == "q-learning" else 1.0)
    )

    if isinstance(controller, QLearningController):
        controller.min_green = min_green
        controller.max_green = max_green
        controller.yellow_duration = yellow_dur
        controller.all_red_duration = all_red_dur
        controller.decision_interval = dec_interval

    cfg = get_scenario_config(
        dataset_name=args.dataset,
        gui=args.gui,
        gui_delay=args.gui_delay,
        step_length=args.step_len,
        yellow_duration=yellow_dur,
        all_red_duration=all_red_dur,
        min_green=min_green,
        max_green=max_green,
        target_tls_id=args.target_tls or getattr(controller, "target_tls_id", None),
        max_steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        control_mode=mode,
        use_tripinfo=args.tripinfo,
        decision_interval=dec_interval,
        use_libsumo=args.use_libsumo,
        save_run_details=args.save_details,
    )

    print(f"[INFO] Initializing simulation: dataset='{args.dataset}', mode='{cfg.control_mode}', controller='{getattr(controller, 'name', 'None')}'")
    runner = SimulationRunner(cfg, controller=controller)
    summary = runner.run(num_steps=args.steps)

    print("\n================ SIMULATION RESULTS ================")
    print(json.dumps(summary, indent=2))
    if summary.get("output_json"):
        print(f"\n[SUCCESS] Results exported to: '{summary.get('output_json')}'")


if __name__ == "__main__":
    main()
