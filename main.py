"""Command-line interface for the Traffic Signal Control simulation platform."""

import argparse
import json
import os
import sys

from src.config import get_scenario_config
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
        choices=["none", "fixed-time", "max-pressure", "q-learning", "dqn"],
        help="Traffic controller algorithm: 'fixed-time', 'max-pressure', 'q-learning', 'dqn', or 'none' (uses --mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint for Q-Learning (.json) or DQN (.pt)",
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
        help="All-red clearance duration in seconds (default: checkpoint value or 0.0)",
    )
    parser.add_argument(
        "--min-green",
        type=float,
        default=None,
        help="Minimum green time constraint in seconds (default: checkpoint value or 5.0)",
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
        help="Allow running RL controllers (q-learning, dqn) without a trained checkpoint",
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
    if args.controller == "fixed-time":
        controller = FixedTimeController(default_green_time=args.fixed_time_green)
        effective_mode = "agent"
    elif args.controller == "max-pressure":
        effective_min_green = args.min_green if args.min_green is not None else 5.0
        controller = MaxPressureController(min_green=effective_min_green, pressure_metric=args.pressure_metric)
        effective_mode = "agent"
    elif args.controller == "q-learning":
        from src.runner import extract_green_phases
        info = inspect_scenario(args.dataset, data_dir="data")
        tl_id = info.tls_ids[0]
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
        effective_mode = "agent"
    elif args.controller == "dqn":
        try:
            from src.controllers.dqn import DQNController
            from src.rl.observations import ObservationBuilder
        except ImportError as exc:
            print(
                f"[ERROR] PyTorch is required to run DQN controller, but failed to import: {exc}\n"
                "Please activate the virtual environment where PyTorch is installed:\n"
                "  .\\.venv\\Scripts\\Activate.ps1\n"
                "Or run directly using:\n"
                "  .\\.venv\\Scripts\\python main.py --controller dqn ...\n"
                "Or install torch via: pip install torch"
            )
            sys.exit(1)

        from src.runner import extract_green_phases
        info = inspect_scenario(args.dataset, data_dir="data")
        tl_id = info.tls_ids[0]
        green_phases = extract_green_phases(args.dataset, tl_id, info.phases_per_tls[tl_id])
        in_lanes = info.controlled_lanes_per_tls.get(tl_id, [])
        model_path = args.model or f"models/dqn/dqn_{args.dataset}_best.pt"
        if os.path.isfile(model_path):
            from src.evaluation.evaluate_rl import load_checkpoint_env_params
            ckpt_params = load_checkpoint_env_params(model_path)
        effective_min_green = args.min_green if args.min_green is not None else ckpt_params.get("min_green", 5.0)
        builder = ObservationBuilder(incoming_lanes=in_lanes, num_green_phases=len(green_phases), min_green=effective_min_green)
        controller = DQNController(
            observation_dim=builder.observation_dim,
            action_dim=len(green_phases),
            obs_builder=builder,
            target_tls_id=tl_id,
            seed=args.seed,
            eval_mode=True,
        )
        if os.path.isfile(model_path):
            controller.load(model_path)
            print(f"[INFO] Loaded DQN model from '{model_path}'")
        else:
            if not args.allow_untrained:
                print(
                    f"[ERROR] DQN model checkpoint '{model_path}' not found!\n"
                    "To evaluate with an untrained agent, explicitly pass '--allow-untrained'."
                )
                sys.exit(1)
            print(f"[WARN] DQN model '{model_path}' not found, using untrained agent (--allow-untrained supplied).")
        effective_mode = "agent"
    else:
        effective_mode = args.mode

    effective_min_green = args.min_green if args.min_green is not None else ckpt_params.get("min_green", 5.0)
    effective_yellow_dur = args.yellow_dur if args.yellow_dur is not None else ckpt_params.get("yellow_duration", 3.0)
    effective_all_red_dur = args.all_red_dur if args.all_red_dur is not None else ckpt_params.get("all_red_duration", 0.0)
    ctrl_norm = (args.controller or "").lower().replace("-", "_")
    effective_dec_interval = (
        args.decision_interval
        if args.decision_interval is not None
        else ckpt_params.get("decision_interval", 5.0 if ctrl_norm in ("dqn", "q_learning") else 1.0)
    )

    print(f"\n[INFO] Initializing simulation for dataset: '{args.dataset}'")
    ctrl_desc = f"Controller: '{controller.name}'" if controller else f"Control Mode: '{effective_mode}'"
    print(f"[INFO] Mode: {'GUI (sumo-gui)' if args.gui else 'Headless (sumo)'} | {ctrl_desc}")

    print(
        f"[INFO] Min Green: {effective_min_green}s | Yellow: {effective_yellow_dur}s | "
        f"All-Red: {effective_all_red_dur}s | Step Length: {args.step_len}s | "
        f"Decision Interval: {effective_dec_interval}s"
    )

    cfg = get_scenario_config(
        dataset_name=args.dataset,
        gui=args.gui,
        step_length=args.step_len,
        yellow_duration=effective_yellow_dur,
        all_red_duration=effective_all_red_dur,
        min_green=effective_min_green,
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

    print("\n================ SIMULATION RESULTS ================")
    print(json.dumps(summary, indent=2))
    print(f"\n[SUCCESS] Results successfully exported to: '{summary.get('output_json')}'")


if __name__ == "__main__":
    main()
