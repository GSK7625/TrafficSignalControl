"""Command-line interface for the Traffic Signal Control simulation platform.

Giao diện dòng lệnh (CLI):
- Không chạy trực tiếp không tham số. Bắt buộc truyền cờ qua dòng lệnh.
- Các cờ lệnh chính: --algorithm (-a), --dataset (-d), --gui, --steps (-s).
"""

import argparse
import json
import sys

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

from src.config import DEFAULT_SAFETY_CONFIG, get_scenario_config
from src.controllers import make_controller
from src.data_inspector import inspect_scenario
from src.runner import SimulationRunner


# Cấu hình mặc định môi trường CLI
DEFAULT_DATASET = "cologne1"
DEFAULT_GUI_DELAY = 50.0
RANDOM_SEED = 42
OUTPUT_DIR = "results"


def parse_args():
    """Phân tích các đối số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="SUMO Traffic Signal Control Platform (CLI)",
        epilog="Example: python main.py -a max-pressure --gui",
    )
    parser.add_argument(
        "--algorithm", "-a",
        required=True,
        choices=["fixed-time", "max-pressure", "q-learning", "none"],
        help="Traffic control algorithm (fixed-time, max-pressure, q-learning, none)",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=DEFAULT_DATASET,
        choices=["cologne1", "ingolstadt1", "cologne3"],
        help=f"Scenario dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Enable SUMO-GUI visual simulation (default: headless)",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=None,
        help="Max simulation steps (default: run entire scenario)",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    algorithm = args.algorithm

    info = inspect_scenario(dataset, data_dir="data")
    dec_interval = 5.0 if algorithm == "q-learning" else 1.0

    controller, mode = make_controller(
        algorithm=algorithm,
        dataset_name=dataset,
        scenario_info=info,
        seed=RANDOM_SEED,
        decision_interval=dec_interval,
        **DEFAULT_SAFETY_CONFIG,
    )

    cfg = get_scenario_config(
        dataset_name=dataset,
        gui=args.gui,
        gui_delay=DEFAULT_GUI_DELAY if args.gui else 0.0,
        step_length=1.0,
        **DEFAULT_SAFETY_CONFIG,
        scenario_info=info,
        target_tls_id=getattr(controller, "target_tls_id", None),
        max_steps=args.steps,
        seed=RANDOM_SEED,
        output_dir=OUTPUT_DIR,
        control_mode=mode,
        use_tripinfo=True,
        decision_interval=dec_interval,
        use_libsumo=True,
        save_run_details=False,
    )

    algo_name = getattr(controller, "name", "Original/Fixed")
    print(f"[INFO] Initializing simulation: dataset='{dataset}', algorithm='{algo_name}', GUI={args.gui}")

    runner = SimulationRunner(cfg, controller=controller)
    summary = runner.run(num_steps=args.steps)

    print("\n================ SIMULATION RESULTS ================")
    print(json.dumps(summary, indent=2))
    if summary.get("output_json"):
        print(f"\n[SUCCESS] Summary report saved to: '{summary.get('output_json')}'")


if __name__ == "__main__":
    main()
