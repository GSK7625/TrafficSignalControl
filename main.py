"""Command-line interface for the Traffic Signal Control simulation platform.

Giao diện dòng lệnh (CLI):
- Không chạy trực tiếp không tham số. Bắt buộc truyền cờ qua dòng lệnh.
- Các cờ lệnh chính: --algorithm (-a), --dataset (-d), --gui, --steps (-s).
"""

import argparse
import json
import sys

# Ensure UTF-8 output and error encoding across Windows consoles
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

from src.evaluation.experiment_runner import ExperimentConfig, run_single_experiment


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
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to Q-Learning model checkpoint (default: models/q_learning/q_learning_<dataset>_best.json)",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def main():
    args = parse_args()
    record, summary = run_single_experiment(ExperimentConfig(
        scenario_name=args.dataset, algorithm=args.algorithm, steps=args.steps,
        model_path=args.model,
        gui=args.gui, gui_delay=DEFAULT_GUI_DELAY if args.gui else 0.0,
        seed=RANDOM_SEED, output_dir=OUTPUT_DIR,
    ), include_summary=True)
    if record["status"] == "FAILED":
        err = record["error_message"]
        if "Model file not found" in err:
            err += (
                f"\nGợi ý: Kịch bản '{args.dataset}' chưa có checkpoint sẵn. "
                f"Hãy chạy huấn luyện trước bằng lệnh:\n"
                f"  python -m src.training.train_q_learning --dataset {args.dataset}\n"
                f"hoặc chỉ định đường dẫn model qua cờ '--model' / '-m'."
            )
        raise RuntimeError(err)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
