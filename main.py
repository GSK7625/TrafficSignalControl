"""Command-line interface for the Traffic Signal Control simulation platform.

Giao diện dòng lệnh (CLI):
- Mặc định đọc toàn bộ cấu hình từ file YAML (mặc định: configs/config.yaml).
- Cho phép chạy trực tiếp không cần truyền cờ: `python main.py`.
- Các cờ lệnh (nếu truyền) sẽ ghi đè cấu hình trong YAML:
  --config (-c), --algorithm (-a), --dataset (-d), --gui, --steps (-s), --model (-m).
"""

import argparse
import json
import os
import sys

# Ensure UTF-8 output and error encoding across Windows consoles
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

from src.config import load_yaml_config, resolve_config_path
from src.evaluation.experiment_runner import ExperimentConfig, run_single_experiment


# Cấu hình mặc định môi trường CLI
DEFAULT_CONFIG_PATH = "configs/config.yaml"
DEFAULT_DATASET = "cologne1"
DEFAULT_GUI_DELAY = 50.0
RANDOM_SEED = 42
OUTPUT_DIR = "results"


def parse_args():
    """Phân tích các đối số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="SUMO Traffic Signal Control Platform (CLI)",
        epilog="Example: python main.py (reads configs/config.yaml) or python main.py -a max-pressure --gui",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--algorithm", "-a",
        default=None,
        choices=["fixed-time", "max-pressure", "q-learning", "none"],
        help="Traffic control algorithm (override YAML config)",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        choices=["cologne1", "ingolstadt1", "cologne3"],
        help="Scenario dataset name (override YAML config)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=None,
        help="Enable SUMO-GUI visual simulation (override YAML config)",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=None,
        help="Max simulation steps (override YAML config)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to Q-Learning model checkpoint (override YAML config)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Bắt buộc phải có file cấu hình YAML để chạy
    config_path = resolve_config_path(args.config)

    yaml_data = load_yaml_config(config_path)
    sim_cfg = yaml_data.get("simulation", {})
    safety_cfg = yaml_data.get("safety", {})
    ql_cfg = yaml_data.get("q_learning", {})

    # Ánh xạ tên cờ CLI sang tên thuộc tính của ExperimentConfig
    cli_key_map = {"dataset": "scenario_name", "model": "model_path"}
    cli_overrides = {
        cli_key_map.get(k, k): v
        for k, v in vars(args).items()
        if v is not None and k != "config"
    }

    base_cfg = {
        "scenario_name": sim_cfg.get("dataset", DEFAULT_DATASET),
        "algorithm": sim_cfg.get("algorithm"),
        "gui": sim_cfg.get("gui", False),
        "steps": sim_cfg.get("steps", None),
        "model_path": sim_cfg.get("model_path", ql_cfg.get("model_path", None)),
        "seed": int(sim_cfg.get("seed", RANDOM_SEED)),
        "output_dir": str(sim_cfg.get("output_dir", OUTPUT_DIR)),
        "step_length": float(sim_cfg.get("step_length", 1.0)),
        "use_tripinfo": bool(sim_cfg.get("use_tripinfo", True)),
    }
    final_cfg = {**base_cfg, **cli_overrides}

    if not final_cfg.get("algorithm"):
        raise ValueError(
            f"Chưa xác định thuật toán điều khiển (algorithm). "
            f"Vui lòng cấu hình 'simulation.algorithm' trong {config_path} hoặc truyền qua cờ -a."
        )

    gui = final_cfg["gui"]
    gui_delay = float(sim_cfg.get("gui_delay", DEFAULT_GUI_DELAY)) if gui else 0.0

    timing_overrides = {
        key: safety_cfg[key]
        for key in ("yellow_duration", "all_red_duration", "min_green", "max_green", "max_red")
        if key in safety_cfg
    }

    exp_config = ExperimentConfig(
        **final_cfg,
        gui_delay=gui_delay,
        timing_overrides=timing_overrides,
    )

    record, summary = run_single_experiment(exp_config, include_summary=True)
    if record["status"] == "FAILED":
        err = record["error_message"]
        if "Model file not found" in err:
            err += (
                f"\nGợi ý: Kịch bản '{exp_config.scenario_name}' chưa có checkpoint sẵn. "
                f"Hãy chạy huấn luyện trước bằng lệnh:\n"
                f"  python -m src.training.train_q_learning --dataset {exp_config.scenario_name}\n"
                f"hoặc chỉ định đường dẫn model qua cờ '--model' / '-m'."
            )
        raise RuntimeError(err)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
