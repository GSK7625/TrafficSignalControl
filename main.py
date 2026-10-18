"""Command-line interface for the Traffic Signal Control simulation platform."""

import argparse
import json
import os
import sys

from src.config import DEFAULT_SAFETY_CONFIG, get_scenario_config
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.controllers.q_learning import QLearningController
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
        "--gui-delay",
        type=float,
        default=0.0,
        help="Delay in milliseconds between simulation steps in SUMO-GUI (e.g. 50 or 100) to observe traffic comfortably (default: 0.0)",
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
        "--algorithm",
        dest="controller",
        type=str,
        default="none",
        choices=["none", "fixed-time", "max-pressure", "q-learning"],
        help="Traffic controller algorithm: 'fixed-time', 'max-pressure', 'q-learning', or 'none' (uses --mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint for Q-Learning (.json)",
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
        default=None,
        help="Yellow clearance duration in seconds (default: checkpoint value or 3.0)",
    )
    parser.add_argument(
        "--all-red-dur",
        type=float,
        default=None,
        help="All-red clearance duration in seconds (default: checkpoint value or 1.0)",
    )
    parser.add_argument(
        "--min-green",
        type=float,
        default=None,
        help="Minimum green time constraint in seconds (default: checkpoint value or 5.0)",
    )
    parser.add_argument(
        "--max-green",
        type=float,
        default=None,
        help="Maximum green time constraint in seconds (default: checkpoint value or 60.0)",
    )
    parser.add_argument(
        "--step-len",
        type=float,
        default=1.0,
        help="Simulation time step in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--decision-interval",
        type=float,
        default=None,
        help="Cadence in seconds between controller decisions (default: 5.0 for RL controllers, 1.0 for baselines)",
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
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Allow running Q-Learning controller without a trained checkpoint",
    )
    parser.add_argument(
        "--target-tls",
        type=str,
        default=None,
        help="Target traffic light ID to control for single-agent RL in multi-intersection networks",
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
    ckpt_params = {}
    if args.target_tls:
        info = inspect_scenario(args.dataset, data_dir="data")
        if args.target_tls not in info.tls_ids:
            parser.error(f"Target TLS '{args.target_tls}' not found in scenario '{args.dataset}'. Available: {info.tls_ids}")
    if args.controller == "fixed-time":
        controller = FixedTimeController(default_green_time=args.fixed_time_green)
        effective_mode = "agent"
    elif args.controller == "max-pressure":
        effective_min_green = args.min_green if args.min_green is not None else 5.0
        controller = MaxPressureController(
            min_green=effective_min_green,
            max_green=args.max_green if args.max_green is not None else DEFAULT_SAFETY_CONFIG["max_green"],
            pressure_metric=args.pressure_metric,
        )
        effective_mode = "agent"
    elif args.controller == "q-learning":
        from src.runner import extract_green_phases
        info = inspect_scenario(args.dataset, data_dir="data")
        if info.tls_count > 1 and not args.target_tls:
            print(
                f"[ERROR] Scenario '{args.dataset}' contains {info.tls_count} traffic light intersections ({info.tls_ids}).\n"
                f"Single-agent Tabular Q-Learning requires a single-intersection scenario (e.g. cologne1, ingolstadt1) "
                f"or an explicit target intersection via '--target-tls <tls_id>'."
            )
            sys.exit(1)
        tl_id = args.target_tls if args.target_tls else info.tls_ids[0]
        if tl_id not in info.phases_per_tls:
            print(f"[ERROR] Target TLS '{tl_id}' not found in scenario '{args.dataset}'. Available: {info.tls_ids}")
            sys.exit(1)
        green_phases = extract_green_phases(args.dataset, tl_id, info.phases_per_tls[tl_id])
        controller = QLearningController(
            num_actions=len(green_phases),
            target_tls_id=tl_id,
            seed=args.seed,
            eval_mode=True,
        )
        model_path = args.model or f"models/q_learning/q_learning_{args.dataset}_best.json"
        if os.path.isfile(model_path):
            from src.evaluation.evaluate_rl import load_checkpoint_env_params
            ckpt_params = load_checkpoint_env_params(model_path)
            if "num_actions" in ckpt_params and ckpt_params["num_actions"] != len(green_phases):
                print(
                    f"[ERROR] Action space mismatch: checkpoint expects {ckpt_params['num_actions']} actions, "
                    f"but target TLS '{tl_id}' in scenario '{args.dataset}' has {len(green_phases)} green phases."
                )
                sys.exit(1)
            if "target_tls_id" in ckpt_params and ckpt_params["target_tls_id"] and ckpt_params["target_tls_id"] != tl_id:
                print(
                    f"[ERROR] Target TLS mismatch: checkpoint targets '{ckpt_params['target_tls_id']}', "
                    f"but CLI selected '{tl_id}'."
                )
                sys.exit(1)
            controller.load(model_path)
            print(f"[INFO] Loaded Q-Learning model from '{model_path}'")
        else:
            if not args.allow_untrained:
                print(
                    f"[ERROR] Q-Learning model checkpoint '{model_path}' not found!\n"
                    "To evaluate with an untrained agent, explicitly pass '--allow-untrained'."
                )
                sys.exit(1)
            print(f"[WARN] Q-Learning model '{model_path}' not found, using untrained agent (--allow-untrained supplied).")
        controller.eval_mode = True
        controller.epsilon = 0.0
        effective_mode = "agent"
    else:
        effective_mode = args.mode

    effective_min_green = args.min_green if args.min_green is not None else ckpt_params.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"])
    effective_max_green = args.max_green if args.max_green is not None else ckpt_params.get("max_green", DEFAULT_SAFETY_CONFIG["max_green"])
    effective_yellow_dur = args.yellow_dur if args.yellow_dur is not None else ckpt_params.get("yellow_duration", DEFAULT_SAFETY_CONFIG["yellow_duration"])
    effective_all_red_dur = args.all_red_dur if args.all_red_dur is not None else ckpt_params.get("all_red_duration", DEFAULT_SAFETY_CONFIG["all_red_duration"])
    ctrl_norm = (args.controller or "").lower().replace("-", "_")
    effective_dec_interval = (
        args.decision_interval
        if args.decision_interval is not None
        else ckpt_params.get("decision_interval", 5.0 if ctrl_norm == "q_learning" else 1.0)
    )

    # CLI timing overrides are explicit and apply to both runner and controller.
    overrides = {}
    for field, value in {
        "min_green": effective_min_green, "max_green": effective_max_green,
        "yellow_duration": effective_yellow_dur, "all_red_duration": effective_all_red_dur,
        "decision_interval": effective_dec_interval,
    }.items():
        if field in ckpt_params and value != ckpt_params[field]:
            overrides[field] = {"checkpoint": ckpt_params[field], "override": value}
        if isinstance(controller, QLearningController):
            setattr(controller, field, value)
    if overrides:
        print(f"[INFO] Configuration overrides: {json.dumps(overrides, sort_keys=True)}")

    print(f"\n[INFO] Initializing simulation for dataset: '{args.dataset}'")
    ctrl_desc = f"Controller: '{controller.name}'" if controller else f"Control Mode: '{effective_mode}'"
    print(f"[INFO] Mode: {'GUI (sumo-gui)' if args.gui else 'Headless (sumo)'} | {ctrl_desc}")

    print(
        f"[INFO] Min Green: {effective_min_green}s | Max Green: {effective_max_green}s | "
        f"Yellow: {effective_yellow_dur}s | All-Red: {effective_all_red_dur}s | "
        f"Step Length: {args.step_len}s | Decision Interval: {effective_dec_interval}s"
    )

    cfg = get_scenario_config(
        dataset_name=args.dataset,
        gui=args.gui,
        gui_delay=args.gui_delay,
        step_length=args.step_len,
        yellow_duration=effective_yellow_dur,
        all_red_duration=effective_all_red_dur,
        min_green=effective_min_green,
        max_green=effective_max_green,
        target_tls_id=args.target_tls or getattr(controller, "target_tls_id", None),
        max_steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        control_mode=effective_mode,
        use_tripinfo=args.tripinfo,
        decision_interval=effective_dec_interval,
    )

    runner = SimulationRunner(cfg, controller=controller)
    print(f"[INFO] Traffic lights found: {list(runner.tls_managers.keys())}")
    for tls_id, mgr in runner.tls_managers.items():
        print(f"  TLS '{tls_id}': {len(mgr.green_phases)} active green phases configured")

    print("[INFO] Starting simulation...")
    summary = runner.run(num_steps=args.steps)

    # Preserve the explicit precedence record alongside the real CLI output.
    summary["configuration_overrides"] = overrides
    if summary.get("output_json"):
        with open(summary["output_json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\n================ SIMULATION RESULTS ================")
    print(json.dumps(summary, indent=2))
    print(f"\n[SUCCESS] Results successfully exported to: '{summary.get('output_json')}'")


if __name__ == "__main__":
    main()
