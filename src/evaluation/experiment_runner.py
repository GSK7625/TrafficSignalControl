"""Automated Experiment Runner for Traffic Signal Control Benchmarks.

Executes, aggregates, and benchmarks traffic signal control algorithms
(Fixed-Time, Max-Pressure, Q-Learning) in a clean, straightforward workflow.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import get_scenario_config
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.controllers.q_learning import QLearningController
from src.data_inspector import inspect_scenario
from src.rl.environment import TrafficSignalRLEnv
from src.runner import SimulationRunner
from src.evaluation.evaluate_rl import load_checkpoint_env_params


@dataclass
class ExperimentConfig:
    """Configuration for a single benchmark evaluation run."""
    scenario_name: str
    algorithm: str  # 'Fixed-Time', 'Max-Pressure', 'Q-Learning'
    seed: int = 42
    training_seed: int = 42
    steps: Optional[int] = None
    model_path: Optional[str] = None
    output_dir: str = "results/evaluation"
    yellow_duration: float = 3.0
    all_red_duration: float = 1.0
    min_green: float = 5.0
    max_green: Optional[float] = 60.0
    fixed_time_green: float = 30.0
    pressure_metric: str = "queue"
    decision_interval: Optional[float] = None
    save_run_details: bool = False


def load_experiments_yaml(config_path: str = "configs/experiments.yaml") -> Dict[str, Any]:
    """Loads benchmark parameters from experiments YAML file if present."""
    if os.path.isfile(config_path):
        try:
            from src.config import load_yaml_config
            return load_yaml_config(config_path)
        except Exception as e:
            print(f"[WARN] Failed to load experiments YAML from '{config_path}': {e}")
    return {}


def resolve_effective_decision_interval(
    cfg: ExperimentConfig, ckpt_params: Optional[Dict[str, Any]] = None
) -> float:
    """Resolves decision interval: 1.0s for baselines, 5.0s (or checkpoint) for RL."""
    if cfg.decision_interval is not None:
        return float(cfg.decision_interval)

    algo_key = cfg.algorithm.lower().replace(" ", "_").replace("-", "_")
    if algo_key in ("q_learning", "qlearning"):
        if ckpt_params and "decision_interval" in ckpt_params:
            return float(ckpt_params["decision_interval"])
        return 5.0

    return 1.0


def resolve_model_path(cfg: ExperimentConfig) -> Optional[str]:
    """Resolves model checkpoint path for RL controllers if not explicitly set."""
    if cfg.model_path:
        return cfg.model_path
    algo_key = cfg.algorithm.lower().replace(" ", "_").replace("-", "_")
    if algo_key in ("q_learning", "qlearning"):
        best_p = f"models/q_learning/q_learning_{cfg.scenario_name}_best.json"
        if os.path.isfile(best_p):
            return best_p
        final_p = f"models/q_learning/q_learning_{cfg.scenario_name}_final.json"
        if os.path.isfile(final_p):
            return final_p
        return best_p
    return None


def get_sumo_version() -> str:
    """Dynamically determines the installed SUMO version."""
    import re
    try:
        out = subprocess.check_output(["sumo", "--version"], text=True, stderr=subprocess.STDOUT)
        first_line = out.strip().splitlines()[0] if out.strip() else ""
        match = re.search(r"(?:sumo(?:\.exe)?|version)\s+([vV]?\d+[\.\d\w_-]*)", first_line, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+\.\d+(?:\.\d+)?[\w_-]*)", first_line)
        if match:
            return match.group(1).lstrip("vV")
    except Exception:
        pass
    try:
        import traci
        if hasattr(traci, "isLoaded") and traci.isLoaded():
            ver = traci.getVersion()
            return f"{ver[0]}.{ver[1]}"
    except Exception:
        pass
    return "1.26.0"


def compute_run_fingerprint(cfg: ExperimentConfig, checkpoint_path: Optional[str] = None, root: Optional[str] = None) -> str:
    """Returns a simple unique run identifier string."""
    return f"{cfg.scenario_name}_{cfg.algorithm}_{cfg.seed}"


def compute_cohort_fingerprint(cfg: ExperimentConfig, checkpoint_path: Optional[str] = None, root: Optional[str] = None) -> str:
    """Returns a simple cohort identifier string."""
    return f"{cfg.scenario_name}_{cfg.algorithm}"


def compute_improvement(baseline_val: float, algo_val: float, higher_is_better: bool = False) -> float:
    """Calculates relative improvement percentage against a baseline."""
    if math.isnan(baseline_val) or math.isnan(algo_val) or math.isinf(baseline_val) or math.isinf(algo_val):
        return 0.0

    if abs(baseline_val) < 1e-9:
        if abs(algo_val) < 1e-9:
            return 0.0
        return (100.0 if algo_val > 0 else -100.0) if higher_is_better else (-100.0 * algo_val if algo_val > 0 else 0.0)

    if higher_is_better:
        return ((algo_val - baseline_val) / abs(baseline_val)) * 100.0
    return ((baseline_val - algo_val) / abs(baseline_val)) * 100.0


def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Executes a single experiment run for any controller algorithm."""
    scenario_info = inspect_scenario(cfg.scenario_name, data_dir="data")
    total_demand = scenario_info.trips_or_vehicles_count

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.scenario_name}_{cfg.algorithm.lower().replace(' ', '_').replace('-', '_')}_s{cfg.seed}_{timestamp}"
    run_dir = os.path.join(cfg.output_dir, "runs", run_id) if cfg.save_run_details else None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)

    print(f"\n---> Running [{cfg.scenario_name.upper()}] | Algorithm: [{cfg.algorithm}] | Seed: {cfg.seed}")

    start_wall_clock = time.time()
    algo_key = cfg.algorithm.lower().replace(" ", "_").replace("-", "_")

    summary: Dict[str, Any] = {}
    episode_reward: Optional[float] = None
    env: Optional[TrafficSignalRLEnv] = None
    runner: Optional[SimulationRunner] = None
    model_path = resolve_model_path(cfg)

    try:
        if algo_key in ("q_learning", "qlearning"):
            ckpt_params = load_checkpoint_env_params(model_path) if model_path else {}
            dec_int = resolve_effective_decision_interval(cfg, ckpt_params)
            m_green = float(cfg.min_green if cfg.min_green is not None else ckpt_params.get("min_green", 5.0))
            mx_green = cfg.max_green if cfg.max_green is not None else ckpt_params.get("max_green", 60.0)
            y_dur = float(cfg.yellow_duration if cfg.yellow_duration is not None else ckpt_params.get("yellow_duration", 3.0))
            ar_dur = float(cfg.all_red_duration if cfg.all_red_duration is not None else ckpt_params.get("all_red_duration", 1.0))

            env_kwargs: Dict[str, Any] = {
                "scenario_name": cfg.scenario_name,
                "decision_interval": dec_int,
                "min_green": m_green,
                "max_green": mx_green,
                "yellow_duration": y_dur,
                "all_red_duration": ar_dur,
                "max_steps": cfg.steps,
                "seed": cfg.seed,
                "output_dir": run_dir,
                "save_run_details": cfg.save_run_details,
                "use_tripinfo": True,
                "gui": False,
            }
            if "state_discretizer" in ckpt_params:
                env_kwargs["state_discretizer"] = ckpt_params["state_discretizer"]

            env = TrafficSignalRLEnv(**env_kwargs)
            agent = QLearningController(
                num_actions=env.num_actions,
                discretizer=env.state_discretizer,
                min_green=m_green,
                max_green=mx_green,
                yellow_duration=y_dur,
                all_red_duration=ar_dur,
                target_tls_id=env.tls_id,
                seed=cfg.seed,
                eval_mode=True,
                tie_break_strategy="first",
            )
            if model_path and os.path.isfile(model_path):
                agent.load(model_path)
            agent.decision_interval = dec_int
            agent.eval_mode = True
            agent.epsilon = 0.0

            _, info = env.reset(seed=cfg.seed)
            valid_actions = info.get("valid_actions")
            discrete_state = info.get("discrete_state")
            accumulated_reward = 0.0

            while True:
                act = agent.select_action_for_state(
                    discrete_state, valid_actions=valid_actions, current_phase=discrete_state[0] if discrete_state else 0
                )
                _, reward, terminated, truncated, next_info = env.step(act)
                accumulated_reward += reward
                valid_actions = next_info.get("valid_actions")
                discrete_state = next_info.get("discrete_state")
                if terminated or truncated:
                    break

            episode_reward = round(accumulated_reward, 2)
            summary = env.get_summary() or {}

        else:
            # Baseline controllers: Fixed-Time, Max-Pressure
            if cfg.algorithm == "Fixed-Time":
                controller = FixedTimeController(default_green_time=cfg.fixed_time_green)
            elif cfg.algorithm == "Max-Pressure":
                controller = MaxPressureController(
                    min_green=cfg.min_green,
                    max_green=cfg.max_green,
                    pressure_metric=cfg.pressure_metric,
                )
            else:
                raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

            sim_cfg = get_scenario_config(
                dataset_name=cfg.scenario_name,
                gui=False,
                step_length=1.0,
                yellow_duration=cfg.yellow_duration,
                all_red_duration=cfg.all_red_duration,
                min_green=cfg.min_green,
                max_green=cfg.max_green,
                max_steps=cfg.steps,
                seed=cfg.seed,
                output_dir=run_dir,
                control_mode="agent",
                use_tripinfo=True,
                decision_interval=resolve_effective_decision_interval(cfg),
                save_run_details=cfg.save_run_details,
            )
            runner = SimulationRunner(sim_cfg, controller=controller)
            summary = runner.run(num_steps=cfg.steps)

    except Exception as e:
        wall_clock = time.time() - start_wall_clock
        print(f"[ERROR] Simulation run failed: {e}")
        return {
            "run_id": run_id,
            "scenario_name": cfg.scenario_name,
            "algorithm": cfg.algorithm,
            "seed": cfg.seed,
            "status": "FAILED",
            "is_valid": False,
            "error_message": str(e),
            "wall_clock_time_seconds": round(wall_clock, 2),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if runner is not None and getattr(runner, "is_running", False):
            try:
                runner.close()
            except Exception:
                pass

    wall_clock = time.time() - start_wall_clock

    # Extract key metrics cleanly
    completed = summary.get("total_completed_vehicles", 0)
    scheduled_demand = summary.get("scheduled_demand_in_window") or total_demand
    completion_rate = round((completed / scheduled_demand * 100.0) if scheduled_demand > 0 else 0.0, 2)
    duration = summary.get("observation_window_duration") or (summary.get("total_steps", 0) * summary.get("step_length", 1.0))
    throughput_vph = round(completed * 3600.0 / duration, 1) if duration > 0 else 0.0

    col_count = summary.get("collision_count", 0) or 0
    tele_count = summary.get("teleport_count", 0) or 0
    safety_status = "INCIDENTS_DETECTED" if (col_count > 0 or tele_count > 0) else "SAFE"
    service_status = "COMPLETED" if completion_rate >= 95.0 else "LOW_COMPLETION"

    mean_wait = summary.get("mean_waiting_time", 0.0)
    trip_wait = summary.get("tripinfo_mean_waiting_time", mean_wait)

    run_record: Dict[str, Any] = {
        "run_id": run_id,
        "scenario_name": cfg.scenario_name,
        "algorithm": cfg.algorithm,
        "seed": cfg.seed,
        "training_seed": cfg.training_seed,
        "checkpoint_path": model_path,
        "status": "SUCCESS",
        "is_valid": True,
        "total_steps": summary.get("total_steps", 0),
        "total_demand": scheduled_demand,
        "total_completed_vehicles": completed,
        "completion_rate_pct": completion_rate,
        "throughput_vph": throughput_vph,
        "mean_waiting_time": mean_wait,
        "tripinfo_mean_waiting_time": trip_wait,
        "system_mean_delay": summary.get("system_mean_delay", mean_wait),
        "system_total_delay": summary.get("system_total_delay", summary.get("total_waiting_time", 0.0)),
        "mean_queue_length": summary.get("mean_queue_length", 0.0),
        "max_queue_length": summary.get("max_queue_length", 0),
        "mean_travel_time": summary.get("mean_travel_time", 0.0),
        "tripinfo_mean_travel_time": summary.get("tripinfo_mean_travel_time", summary.get("mean_travel_time", 0.0)),
        "total_time_loss": summary.get("total_time_loss", 0.0),
        "tripinfo_total_time_loss": summary.get("tripinfo_total_time_loss", summary.get("total_time_loss", 0.0)),
        "collision_count": col_count,
        "teleport_count": tele_count,
        "phase_change_count": summary.get("phase_change_count", 0),
        "episode_reward": episode_reward,
        "safety_status": safety_status,
        "service_status": service_status,
        "wall_clock_time_seconds": round(wall_clock, 2),
        "output_dir": run_dir,
    }

    # Save summary JSON in run directory if requested
    if cfg.save_run_details and run_dir:
        summary_path = os.path.join(run_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)

    print(
        f"==> Completed in {wall_clock:.1f}s | "
        f"Vehicles: {completed}/{scheduled_demand} ({completion_rate:.1f}%) | "
        f"Wait: {trip_wait:.2f}s | Queue: {run_record['mean_queue_length']:.2f} | "
        f"Collisions: {col_count}"
    )
    return run_record


def aggregate_benchmark_statistics(
    run_records: List[Dict[str, Any]],
    baseline_algo: str = "Fixed-Time",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Computes mean, std, and improvement percentages grouped by scenario and algorithm."""
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    failed_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for r in run_records:
        scen = r.get("scenario_name", "unknown")
        algo = r.get("algorithm", "unknown")
        if not r.get("is_valid", False) or r.get("status") == "FAILED":
            failed_groups.setdefault(scen, {}).setdefault(algo, []).append(r)
        else:
            groups.setdefault(scen, {}).setdefault(algo, []).append(r)

    metrics_keys = [
        ("tripinfo_mean_waiting_time", False),
        ("mean_waiting_time", False),
        ("mean_queue_length", False),
        ("total_completed_vehicles", True),
        ("completion_rate_pct", True),
        ("throughput_vph", True),
        ("tripinfo_mean_travel_time", False),
        ("tripinfo_total_time_loss", False),
        ("system_mean_delay", False),
        ("system_total_delay", False),
        ("collision_count", False),
        ("teleport_count", False),
        ("phase_change_count", False),
    ]

    summary_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_scenarios = sorted(list(set(list(groups.keys()) + list(failed_groups.keys()))))
    for scen in all_scenarios:
        summary_stats[scen] = {}

    for scen, algos in groups.items():
        for algo, runs in algos.items():
            distinct_seeds = sorted(list({r["seed"] for r in runs if "seed" in r}))
            algo_stats: Dict[str, Any] = {
                "run_count": len(runs),
                "seeds": distinct_seeds,
                "status": "SUCCESS",
                "is_valid": True,
            }

            for key, _ in metrics_keys:
                vals = [float(r[key]) for r in runs if r.get(key) is not None]
                if vals:
                    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    entry = {
                        "mean": round(float(np.mean(vals)), 2),
                        "std": round(std_val, 2),
                        "min": round(float(np.min(vals)), 2),
                        "max": round(float(np.max(vals)), 2),
                        "available": True,
                    }
                    if key in ("collision_count", "teleport_count"):
                        entry["sum"] = int(np.sum(vals))
                    algo_stats[key] = entry
                else:
                    entry = {"mean": None, "std": None, "min": None, "max": None, "available": False}
                    if key in ("collision_count", "teleport_count"):
                        entry["sum"] = 0
                    algo_stats[key] = entry

            # Episode rewards (for RL)
            rewards = [r["episode_reward"] for r in runs if r.get("episode_reward") is not None]
            if rewards:
                std_rew = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
                algo_stats["episode_reward"] = {
                    "mean": round(float(np.mean(rewards)), 2),
                    "std": round(std_rew, 2),
                    "min": round(float(np.min(rewards)), 2),
                    "max": round(float(np.max(rewards)), 2),
                }

            col_sum = algo_stats.get("collision_count", {}).get("sum", 0)
            tel_sum = algo_stats.get("teleport_count", {}).get("sum", 0)
            algo_stats["safety_status"] = "INCIDENTS_DETECTED" if (col_sum > 0 or tel_sum > 0) else "SAFE"
            cr_mean = algo_stats.get("completion_rate_pct", {}).get("mean", 100.0)
            algo_stats["service_status"] = "COMPLETED" if (cr_mean is not None and cr_mean >= 95.0) else "LOW_COMPLETION"

            summary_stats[scen][algo] = algo_stats

        # Compute improvements relative to baseline algorithm
        baseline_stats = summary_stats[scen].get(baseline_algo)
        if baseline_stats and baseline_stats.get("is_valid", False):
            for algo, stats in summary_stats[scen].items():
                if algo == baseline_algo or not stats.get("is_valid", False):
                    continue
                improvements: Dict[str, float] = {}
                for key, higher_is_better in metrics_keys:
                    if key in baseline_stats and key in stats:
                        b_mean = baseline_stats[key].get("mean")
                        a_mean = stats[key].get("mean")
                        if b_mean is not None and a_mean is not None:
                            imp = compute_improvement(b_mean, a_mean, higher_is_better=higher_is_better)
                            improvements[f"{key}_improvement_pct"] = round(imp, 2)
                stats["improvements_vs_baseline"] = improvements

    # Record failed algorithms
    for scen, f_algos in failed_groups.items():
        summary_stats.setdefault(scen, {})
        for algo, f_runs in f_algos.items():
            if algo not in summary_stats[scen]:
                err_msgs = [fr.get("error_message", "Simulation failed") for fr in f_runs]
                summary_stats[scen][algo] = {
                    "run_count": len(f_runs),
                    "failed_count": len(f_runs),
                    "seeds": sorted([fr.get("seed", 0) for fr in f_runs]),
                    "status": "FAILED",
                    "is_valid": False,
                    "error_messages": list(dict.fromkeys(err_msgs)),
                }

    return summary_stats


def run_full_benchmark_matrix(
    scenarios: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    config_file: str = "configs/experiments.yaml",
    output_dir: str = "results/evaluation",
    resume: bool = True,
    save_run_details: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs full evaluation benchmark across all scenarios, algorithms, and seeds."""
    yaml_cfg = load_experiments_yaml(config_file)
    sim_yaml = yaml_cfg.get("simulation", {})

    if scenarios is None:
        scenarios = yaml_cfg.get("scenarios", ["cologne1", "ingolstadt1"])
    if algorithms is None:
        algorithms = yaml_cfg.get("algorithms", ["Fixed-Time", "Max-Pressure", "Q-Learning"])
    if seeds is None:
        seeds = [42, 101, 2024]

    default_yellow = float(sim_yaml.get("yellow_duration", 3.0))
    default_all_red = float(sim_yaml.get("all_red_duration", 1.0))
    default_min_green = float(sim_yaml.get("min_green", 5.0))
    default_max_green = float(sim_yaml.get("max_green", 60.0))
    default_steps = int(sim_yaml.get("steps", 3600)) if "steps" in sim_yaml else None

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

    executed_keys = {
        (r.get("scenario_name"), r.get("algorithm"), r.get("seed"))
        for r in all_records
        if r.get("status") == "SUCCESS"
    }

    total_tasks = len(scenarios) * len(algorithms) * len(seeds)
    task_idx = 0

    print(f"\n================================================================================")
    print(f"STARTING BENCHMARK EVALUATION")
    print(f"Scenarios: {scenarios} | Algorithms: {algorithms} | Seeds: {seeds} ({total_tasks} runs)")
    print(f"================================================================================")

    for scen in scenarios:
        for algo in algorithms:
            for s in seeds:
                task_idx += 1
                if resume and (scen, algo, s) in executed_keys:
                    print(f"[{task_idx}/{total_tasks}] Skipping already completed: {scen} | {algo} | Seed {s}")
                    continue

                cfg = ExperimentConfig(
                    scenario_name=scen,
                    algorithm=algo,
                    seed=s,
                    output_dir=output_dir,
                    yellow_duration=default_yellow,
                    all_red_duration=default_all_red,
                    min_green=default_min_green,
                    max_green=default_max_green,
                    steps=default_steps,
                    save_run_details=save_run_details,
                )
                rec = run_single_experiment(cfg)
                all_records.append(rec)

                with open(master_records_file, "w", encoding="utf-8") as f:
                    json.dump(all_records, f, indent=2)

    summary_stats = aggregate_benchmark_statistics(all_records, baseline_algo="Fixed-Time")
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    csv_file = os.path.join(output_dir, "benchmark_runs.csv")
    try:
        import pandas as pd
        df = pd.DataFrame(all_records)
        df.to_csv(csv_file, index=False)
    except Exception:
        pass

    print(f"\n================================================================================")
    print(f"BENCHMARK COMPLETED: Results saved in '{output_dir}'")
    print(f"================================================================================")
    return all_records, summary_stats


ALGO_NAME_MAP = {
    "ft": "Fixed-Time",
    "fixed-time": "Fixed-Time",
    "fixed_time": "Fixed-Time",
    "mp": "Max-Pressure",
    "max-pressure": "Max-Pressure",
    "max_pressure": "Max-Pressure",
    "ql": "Q-Learning",
    "q-learning": "Q-Learning",
    "q_learning": "Q-Learning",
}


def normalize_algo_name(name: str) -> str:
    """Normalizes algorithm names including short aliases like ft, mp, ql."""
    cleaned = name.strip()
    return ALGO_NAME_MAP.get(cleaned.lower(), cleaned)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--scenario", type=str, default="cologne1")
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="Fixed-Time")
    parser.add_argument("--algorithms", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--save-details", action="store_true", help="Save detailed per-run simulation files (XML/CSV) in runs/")
    args = parser.parse_args()

    is_multi = args.run_all or (args.scenarios is not None) or (args.algorithms is not None) or (args.seeds is not None)

    if is_multi:
        scenarios_list = (
            [s.strip() for s in args.scenarios.split(",") if s.strip()]
            if args.scenarios
            else ([args.scenario] if not args.run_all else None)
        )
        algorithms_list = (
            [normalize_algo_name(a) for a in args.algorithms.split(",") if a.strip()]
            if args.algorithms
            else ([normalize_algo_name(args.algorithm)] if not args.run_all else None)
        )
        seeds_list = (
            [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            if args.seeds
            else ([args.seed] if not args.run_all else None)
        )
        run_full_benchmark_matrix(
            scenarios=scenarios_list,
            algorithms=algorithms_list,
            seeds=seeds_list,
            config_file=args.config,
            resume=not args.fresh,
            save_run_details=args.save_details,
        )
    else:
        norm_algo = normalize_algo_name(args.algorithm)
        yaml_cfg = load_experiments_yaml(args.config)
        sim_yaml = yaml_cfg.get("simulation", {})
        cfg = ExperimentConfig(
            scenario_name=args.scenario,
            algorithm=norm_algo,
            seed=args.seed,
            yellow_duration=float(sim_yaml.get("yellow_duration", 3.0)),
            all_red_duration=float(sim_yaml.get("all_red_duration", 1.0)),
            min_green=float(sim_yaml.get("min_green", 5.0)),
            max_green=float(sim_yaml.get("max_green", 60.0)),
            steps=int(sim_yaml.get("steps", 3600)) if "steps" in sim_yaml else None,
            save_run_details=args.save_details,
        )
        run_single_experiment(cfg)
