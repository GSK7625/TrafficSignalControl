"""Automated Experiment Runner for Phase 6 benchmarking.

Executes, validates, and aggregates benchmarks across all 4 traffic signal control
algorithms (Fixed-Time, Max-Pressure, Q-Learning, DQN) plus the Original SUMO baseline
under strictly standardized, reproducible experimental conditions.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
import platform
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import SimulationConfig, get_scenario_config
from src.controllers.dqn import DQNController
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.controllers.q_learning import QLearningController
from src.data_inspector import inspect_scenario
from src.metrics import MetricsCollector
from src.rl.environment import TrafficSignalRLEnv
from src.runner import SimulationRunner, extract_green_phases


# RESCO Known Scenario Ground Truth Demand (total vehicles in .rou.xml)
SCENARIO_DEMAND = {
    "cologne1": 2015,
    "ingolstadt1": 1716,
    "cologne3": 4494,
    "cologne8": 2046,
    "ingolstadt7": 3031,
    "ingolstadt21": 4283,
}


@dataclass
class ExperimentConfig:
    """Configuration for a single benchmark evaluation run."""
    scenario_name: str
    algorithm: str  # 'Fixed-Time', 'Max-Pressure', 'Q-Learning', 'DQN', 'Original SUMO'
    seed: int = 42
    training_seed: int = 42
    steps: Optional[int] = None  # None = full horizon (3600s)
    model_path: Optional[str] = None
    output_dir: str = "results/evaluation"
    yellow_duration: float = 3.0
    all_red_duration: float = 0.0
    min_green: float = 5.0
    fixed_time_green: float = 30.0
    pressure_metric: str = "queue"
    decision_interval: float = 5.0


def compute_improvement(baseline_val: float, algo_val: float, higher_is_better: bool = False) -> float:
    """Calculates relative improvement percentage against a baseline.

    Args:
        baseline_val: Reference baseline metric value.
        algo_val: Evaluated algorithm metric value.
        higher_is_better: False if lower metric indicates better performance (e.g. waiting time, queue),
                          True if higher metric indicates better performance (e.g. throughput).

    Returns:
        Improvement percentage. Positive represents improvement over baseline.
    """
    if math.isnan(baseline_val) or math.isnan(algo_val) or math.isinf(baseline_val) or math.isinf(algo_val):
        return 0.0

    if abs(baseline_val) < 1e-9:
        return 0.0

    if higher_is_better:
        return ((algo_val - baseline_val) / abs(baseline_val)) * 100.0
    else:
        return ((baseline_val - algo_val) / abs(baseline_val)) * 100.0


def validate_run_metrics(metrics: Dict[str, Any], total_demand: int) -> Tuple[bool, List[str]]:
    """Performs rigorous anomaly detection and scientific integrity checks on simulation results.

    Returns:
        Tuple of (is_valid, list_of_anomaly_messages).
    """
    anomalies: List[str] = []

    # Check for NaN / Infinity
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if math.isnan(v):
                anomalies.append(f"Metric '{k}' has NaN value")
            elif math.isinf(v):
                anomalies.append(f"Metric '{k}' has Infinity value")

    # Waiting time checks
    mean_wait = metrics.get("mean_waiting_time", 0.0)
    trip_wait = metrics.get("tripinfo_mean_waiting_time", mean_wait)
    if mean_wait < 0.0 or trip_wait < 0.0:
        anomalies.append(f"Negative waiting time detected: mean={mean_wait}, tripinfo={trip_wait}")

    # Queue length checks
    queue_len = metrics.get("mean_queue_length", 0.0)
    if queue_len < 0.0:
        anomalies.append(f"Negative queue length detected: {queue_len}")

    # Travel time / Time loss checks
    travel_time = metrics.get("mean_travel_time", 0.0)
    if travel_time < 0.0:
        anomalies.append(f"Negative travel time detected: {travel_time}")

    time_loss = metrics.get("total_time_loss", 0.0)
    if time_loss < 0.0:
        anomalies.append(f"Negative total time loss detected: {time_loss}")

    # Completion count & rate checks
    completed = metrics.get("total_completed_vehicles", 0)
    if completed < 0:
        anomalies.append(f"Negative completed vehicles count: {completed}")

    # If full horizon run (total_steps >= 1800), having zero completed vehicles is an anomaly
    total_steps = metrics.get("total_steps", 0)
    if total_steps >= 1800 and completed == 0:
        anomalies.append(f"Abnormal traffic lock: 0 completed vehicles in {total_steps}s")

    if total_demand > 0:
        completion_rate = (completed / total_demand) * 100.0
        if completion_rate > 105.0:  # Allow 5% margin for any synthetic additions
            anomalies.append(f"Completion rate exceeds 100%: {completion_rate:.1f}% ({completed}/{total_demand})")

    is_valid = len(anomalies) == 0
    return is_valid, anomalies


def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Executes a single experiment run under standardized evaluation protocol."""
    scenario_info = inspect_scenario(cfg.scenario_name, data_dir="data")
    total_demand = SCENARIO_DEMAND.get(cfg.scenario_name, scenario_info.trips_or_vehicles_count)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.scenario_name}_{cfg.algorithm.lower().replace(' ', '_').replace('-', '_')}_s{cfg.seed}_{timestamp}"
    run_dir = os.path.join(cfg.output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n================================================================================")
    print(f"EXPERIMENT RUN: [{cfg.scenario_name.upper()}] | Algorithm: [{cfg.algorithm}] | Seed: {cfg.seed}")
    print(f"Output folder: {run_dir}")
    print(f"================================================================================")

    start_wall_clock = time.time()
    algo_key = cfg.algorithm.lower().replace(" ", "_").replace("-", "_")

    model_path = cfg.model_path
    if model_path is None:
        if algo_key in ("q_learning", "qlearning"):
            model_path = f"models/q_learning/q_learning_{cfg.scenario_name}_best.json"
        elif algo_key in ("dqn", "deep_q_network"):
            model_path = f"models/dqn/dqn_{cfg.scenario_name}_best.pt"

    summary: Dict[str, Any] = {}
    episode_reward: Optional[float] = None
    checkpoint_id: Optional[str] = os.path.basename(model_path) if model_path else None

    try:
        if algo_key in ("q_learning", "qlearning", "dqn", "deep_q_network"):
            # RL Agents evaluated in evaluation mode (zero exploration, zero learning updates)
            env = TrafficSignalRLEnv(
                scenario_name=cfg.scenario_name,
                decision_interval=cfg.decision_interval,
                min_green=cfg.min_green,
                yellow_duration=cfg.yellow_duration,
                all_red_duration=cfg.all_red_duration,
                max_steps=cfg.steps,
                seed=cfg.seed,
                output_dir=run_dir,
                use_tripinfo=True,
                gui=False,
            )

            if algo_key in ("q_learning", "qlearning"):
                agent = QLearningController(
                    num_actions=env.num_actions,
                    seed=cfg.seed,
                    eval_mode=True,
                )
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(f"Q-Learning model checkpoint not found: {model_path}")
                agent.load(model_path)
                agent.eval_mode = True
                agent.epsilon = 0.0  # Strict zero exploration
            else:
                agent = DQNController(
                    observation_dim=env.observation_dim,
                    action_dim=env.num_actions,
                    obs_builder=env.obs_builder,
                    seed=cfg.seed,
                    eval_mode=True,
                )
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(f"DQN model checkpoint not found: {model_path}")
                agent.load(model_path)
                agent.eval_mode = True
                agent.epsilon = 0.0  # Strict zero exploration

            obs_vec, info = env.reset(seed=cfg.seed)
            valid_actions = info["valid_actions"]
            discrete_state = info["discrete_state"]

            accumulated_reward = 0.0
            step_count = 0

            while True:
                if algo_key in ("q_learning", "qlearning"):
                    act = agent.select_action_for_state(discrete_state, valid_actions=valid_actions)
                else:
                    act = agent.select_action_for_state(obs_vec, valid_actions=valid_actions)

                next_obs, reward, terminated, truncated, next_info = env.step(act)
                accumulated_reward += reward
                step_count += 1

                obs_vec = next_obs
                valid_actions = next_info["valid_actions"]
                discrete_state = next_info["discrete_state"]

                if terminated or truncated:
                    break

            episode_reward = round(accumulated_reward, 2)
            summary = env.get_summary() or {}
            env.close()

        else:
            # Baseline controllers: Fixed-Time, Max-Pressure, Original SUMO
            controller = None
            if cfg.algorithm == "Fixed-Time":
                controller = FixedTimeController(default_green_time=cfg.fixed_time_green)
                control_mode = "agent"
            elif cfg.algorithm == "Max-Pressure":
                controller = MaxPressureController(min_green=cfg.min_green, pressure_metric=cfg.pressure_metric)
                control_mode = "agent"
            elif cfg.algorithm == "Original SUMO":
                control_mode = "original"
            else:
                raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

            sim_cfg = get_scenario_config(
                dataset_name=cfg.scenario_name,
                gui=False,
                step_length=1.0,
                yellow_duration=cfg.yellow_duration,
                all_red_duration=cfg.all_red_duration,
                min_green=cfg.min_green,
                max_steps=cfg.steps,
                seed=cfg.seed,
                output_dir=run_dir,
                control_mode=control_mode,
                use_tripinfo=True,
            )

            runner = SimulationRunner(sim_cfg, controller=controller)
            summary = runner.run(num_steps=cfg.steps)

    except Exception as e:
        wall_clock = time.time() - start_wall_clock
        err_msg = f"Simulation failed with exception: {str(e)}"
        print(f"[ERROR] {err_msg}")
        return {
            "run_id": run_id,
            "scenario_name": cfg.scenario_name,
            "algorithm": cfg.algorithm,
            "seed": cfg.seed,
            "training_seed": cfg.training_seed,
            "checkpoint_id": checkpoint_id,
            "status": "FAILED",
            "is_valid": False,
            "error_message": err_msg,
            "anomalies": [err_msg],
            "wall_clock_time_seconds": round(wall_clock, 2),
        }

    wall_clock = time.time() - start_wall_clock

    # Extract metrics
    completed = summary.get("total_completed_vehicles", 0)
    completion_rate = round((completed / total_demand * 100.0) if total_demand > 0 else 0.0, 2)
    throughput_vph = round(completed * (3600.0 / max(1, summary.get("total_steps", 3600))), 1)

    is_valid, anomalies = validate_run_metrics(summary, total_demand=total_demand)
    exec_status = "SUCCESS" if (is_valid and summary.get("is_valid", True)) else "INVALID"

    run_record: Dict[str, Any] = {
        "run_id": run_id,
        "scenario_name": cfg.scenario_name,
        "algorithm": cfg.algorithm,
        "seed": cfg.seed,
        "training_seed": cfg.training_seed,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": model_path if model_path else None,
        "status": exec_status,
        "is_valid": is_valid and summary.get("is_valid", True),
        "total_steps": summary.get("total_steps", 0),
        "total_demand": total_demand,
        "total_completed_vehicles": completed,
        "completion_rate_pct": completion_rate,
        "throughput_vph": throughput_vph,
        "mean_waiting_time": summary.get("mean_waiting_time", 0.0),
        "tripinfo_mean_waiting_time": summary.get("tripinfo_mean_waiting_time", summary.get("mean_waiting_time", 0.0)),
        "tripinfo_total_waiting_time": summary.get("tripinfo_total_waiting_time", summary.get("total_waiting_time", 0.0)),
        "mean_queue_length": summary.get("mean_queue_length", 0.0),
        "max_queue_length": summary.get("max_queue_length", 0),
        "mean_travel_time": summary.get("mean_travel_time", 0.0),
        "tripinfo_mean_travel_time": summary.get("tripinfo_mean_travel_time", summary.get("mean_travel_time", 0.0)),
        "total_time_loss": summary.get("total_time_loss", 0.0),
        "tripinfo_total_time_loss": summary.get("tripinfo_total_time_loss", summary.get("total_time_loss", 0.0)),
        "tripinfo_mean_time_loss": summary.get("tripinfo_mean_time_loss", 0.0),
        "running_vehicles_at_end": summary.get("running_vehicles_at_end", 0),
        "pending_insertion_vehicles": summary.get("pending_insertion_vehicles", 0),
        "teleport_count": summary.get("teleport_count", 0),
        "collision_count": summary.get("collision_count", 0),
        "phase_change_count": summary.get("phase_change_count", 0),
        "phase_durations": summary.get("phase_durations_per_tls", {}),
        "episode_reward": episode_reward,
        "anomalies": anomalies,
        "wall_clock_time_seconds": round(wall_clock, 2),
        "sumo_version": "1.28.0",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "output_dir": run_dir,
    }

    # Save summary JSON in run directory
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    print(
        f"==> Status: {exec_status} | Time: {wall_clock:.2f}s | "
        f"Completed: {completed}/{total_demand} ({completion_rate:.1f}%) | "
        f"Mean Wait: {run_record['tripinfo_mean_waiting_time']:.2f}s | "
        f"Mean Queue: {run_record['mean_queue_length']:.2f} | "
        f"Teleports: {run_record['teleport_count']}"
    )
    if anomalies:
        print(f"    [WARNINGS/ANOMALIES]: {anomalies}")

    return run_record


def aggregate_benchmark_statistics(
    run_records: List[Dict[str, Any]],
    baseline_algo: str = "Fixed-Time",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Computes comprehensive mean, std, min, max, and percentage improvements across runs."""
    # Group by (scenario, algorithm)
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in run_records:
        if not r.get("is_valid", False):
            continue
        scen = r["scenario_name"]
        algo = r["algorithm"]
        grouped.setdefault(scen, {}).setdefault(algo, []).append(r)

    metrics_keys = [
        ("tripinfo_mean_waiting_time", False),
        ("mean_queue_length", False),
        ("total_completed_vehicles", True),
        ("completion_rate_pct", True),
        ("throughput_vph", True),
        ("tripinfo_mean_travel_time", False),
        ("tripinfo_total_time_loss", False),
        ("teleport_count", False),
        ("phase_change_count", False),
    ]

    summary_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for scen, algos in grouped.items():
        summary_stats[scen] = {}

        # 1. Compute stats for each algorithm
        for algo, runs in algos.items():
            algo_stats: Dict[str, Any] = {
                "run_count": len(runs),
                "seeds": [r["seed"] for r in runs],
            }
            for key, _ in metrics_keys:
                vals = [float(r.get(key, 0.0)) for r in runs]
                if vals:
                    algo_stats[key] = {
                        "mean": round(float(np.mean(vals)), 2),
                        "std": round(float(np.std(vals)), 2),
                        "min": round(float(np.min(vals)), 2),
                        "max": round(float(np.max(vals)), 2),
                    }
                else:
                    algo_stats[key] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

            # Optional RL reward
            rewards = [r["episode_reward"] for r in runs if r.get("episode_reward") is not None]
            if rewards:
                algo_stats["episode_reward"] = {
                    "mean": round(float(np.mean(rewards)), 2),
                    "std": round(float(np.std(rewards)), 2),
                    "min": round(float(np.min(rewards)), 2),
                    "max": round(float(np.max(rewards)), 2),
                }

            summary_stats[scen][algo] = algo_stats

        # 2. Compute improvements relative to baseline
        baseline_stats = summary_stats[scen].get(baseline_algo)
        if baseline_stats:
            for algo, stats in summary_stats[scen].items():
                improvements: Dict[str, float] = {}
                for key, higher_is_better in metrics_keys:
                    b_mean = baseline_stats[key]["mean"]
                    a_mean = stats[key]["mean"]
                    imp = compute_improvement(b_mean, a_mean, higher_is_better=higher_is_better)
                    improvements[f"{key}_improvement_pct"] = round(imp, 2)
                stats["improvements_vs_baseline"] = improvements

    return summary_stats


def run_full_benchmark_matrix(
    scenarios: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: str = "results/evaluation",
    resume: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs full evaluation benchmark across all scenarios, algorithms, and seeds."""
    if scenarios is None:
        scenarios = ["ingolstadt1", "cologne1"]
    if algorithms is None:
        algorithms = ["Fixed-Time", "Max-Pressure", "Q-Learning", "DQN", "Original SUMO"]
    if seeds is None:
        seeds = [42, 101, 2024]

    os.makedirs(output_dir, exist_ok=True)
    master_records_file = os.path.join(output_dir, "benchmark_runs.json")
    all_records: List[Dict[str, Any]] = []

    if resume and os.path.isfile(master_records_file):
        try:
            with open(master_records_file, "r", encoding="utf-8") as f:
                all_records = json.load(f)
            print(f"[INFO] Resumed {len(all_records)} existing runs from '{master_records_file}'")
        except Exception as e:
            print(f"[WARN] Failed to resume existing records: {e}")

    # Build set of already executed keys: (scenario, algorithm, seed)
    executed_keys = {
        (r["scenario_name"], r["algorithm"], r["seed"])
        for r in all_records
        if r.get("is_valid", False) and r.get("status") == "SUCCESS"
    }

    total_tasks = len(scenarios) * len(algorithms) * len(seeds)
    task_idx = 0

    print(f"\n================================================================================")
    print(f"STARTING COMPREHENSIVE BENCHMARK EVALUATION (PHASE 6)")
    print(f"Scenarios: {scenarios}")
    print(f"Algorithms: {algorithms}")
    print(f"Evaluation Seeds: {seeds} (Total Tasks: {total_tasks})")
    print(f"================================================================================")

    for scen in scenarios:
        for algo in algorithms:
            for s in seeds:
                task_idx += 1
                if resume and (scen, algo, s) in executed_keys:
                    print(f"[{task_idx}/{total_tasks}] Skipping already completed: {scen} | {algo} | Seed {s}")
                    continue

                print(f"\n[{task_idx}/{total_tasks}] Executing: {scen} | {algo} | Seed {s}")
                cfg = ExperimentConfig(
                    scenario_name=scen,
                    algorithm=algo,
                    seed=s,
                    training_seed=42,
                    output_dir=output_dir,
                )
                rec = run_single_experiment(cfg)
                all_records.append(rec)

                # Save incremental master record
                with open(master_records_file, "w", encoding="utf-8") as f:
                    json.dump(all_records, f, indent=2)

    # Compute statistical aggregation
    summary_stats = aggregate_benchmark_statistics(all_records, baseline_algo="Fixed-Time")
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    # Export CSV table of all runs
    csv_file = os.path.join(output_dir, "benchmark_runs.csv")
    import pandas as pd
    df = pd.DataFrame(all_records)
    # Exclude complex nested dicts for CSV export
    cols_to_export = [c for c in df.columns if c not in ("phase_durations", "anomalies")]
    df[cols_to_export].to_csv(csv_file, index=False)

    print(f"\n================================================================================")
    print(f"BENCHMARK COMPLETE!")
    print(f"Master runs saved: '{master_records_file}' and '{csv_file}'")
    print(f"Statistical summary saved: '{summary_file}'")
    print(f"================================================================================")

    return all_records, summary_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6 Experiment Runner")
    parser.add_argument("--scenario", type=str, default="ingolstadt1", help="Scenario name")
    parser.add_argument("--algorithm", type=str, default="Fixed-Time", help="Algorithm name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--run-all", action="store_true", help="Run full benchmark matrix")
    args = parser.parse_args()

    if args.run_all:
        run_full_benchmark_matrix()
    else:
        cfg = ExperimentConfig(scenario_name=args.scenario, algorithm=args.algorithm, seed=args.seed)
        run_single_experiment(cfg)
