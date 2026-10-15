"""Automated Experiment Runner for Phase 6 benchmarking.

Executes, validates, and aggregates benchmarks across all 3 traffic signal control
algorithms (Fixed-Time, Max-Pressure, Q-Learning)
under strictly standardized, reproducible experimental conditions.
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
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
    steps: Optional[int] = None  # None = full horizon (3600s)
    model_path: Optional[str] = None
    output_dir: str = "results/evaluation"
    yellow_duration: float = 3.0
    all_red_duration: float = 1.0  # Matches configs/experiments.yaml
    min_green: float = 5.0
    max_green: Optional[float] = 60.0
    fixed_time_green: float = 30.0
    pressure_metric: str = "queue"
    decision_interval: Optional[float] = None


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
    """Resolves the effective decision interval (cadence) for an evaluation run.

    Baselines (Fixed-Time, Max-Pressure) default to 1.0s to ensure exact phase timings
    and prevent phase green time elongation (e.g. 30s turning into 32s).
    RL algorithms (Q-Learning) default to checkpoint values if present, or 5.0s.
    An explicit cfg.decision_interval always overrides defaults.
    """
    if cfg.decision_interval is not None:
        return float(cfg.decision_interval)

    algo_key = cfg.algorithm.lower().replace(" ", "_").replace("-", "_")
    if algo_key in ("q_learning", "qlearning"):
        if ckpt_params and "decision_interval" in ckpt_params:
            return float(ckpt_params["decision_interval"])
        return 5.0

    return 1.0


def resolve_model_path(cfg: ExperimentConfig) -> Optional[str]:
    """Resolves default model checkpoint path for RL controllers if not explicitly set."""
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
    return "unknown"


def _hash_scenario_inputs(scenario_name: str, hasher) -> None:
    """Hashes scenario network/route files and runtime SUMO version into the fingerprint."""
    scenario_dir = os.path.join("data", scenario_name)
    if os.path.isdir(scenario_dir):
        for fname in sorted(os.listdir(scenario_dir)):
            if fname.endswith((".net.xml", ".rou.xml", ".flo.xml", ".sumocfg")):
                fpath = os.path.join(scenario_dir, fname)
                try:
                    with open(fpath, "rb") as df:
                        hasher.update(df.read())
                except Exception:
                    pass
    hasher.update(get_sumo_version().encode("utf-8"))


def compute_run_fingerprint(cfg: ExperimentConfig, checkpoint_path: Optional[str] = None) -> str:
    """Computes a deterministic SHA-256 fingerprint of the experiment configuration and model artifact."""
    hasher = hashlib.sha256()
    payload = {
        "scenario_name": cfg.scenario_name,
        "algorithm": cfg.algorithm,
        "seed": cfg.seed,
        "training_seed": cfg.training_seed,
        "steps": cfg.steps,
        "yellow_duration": cfg.yellow_duration,
        "all_red_duration": cfg.all_red_duration,
        "min_green": cfg.min_green,
        "max_green": cfg.max_green,
        "fixed_time_green": cfg.fixed_time_green,
        "pressure_metric": cfg.pressure_metric,
        "decision_interval": cfg.decision_interval,
    }
    hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    _hash_scenario_inputs(cfg.scenario_name, hasher)

    target_checkpoint = checkpoint_path or cfg.model_path or resolve_model_path(cfg)
    if target_checkpoint and os.path.isfile(target_checkpoint):
        try:
            with open(target_checkpoint, "rb") as f:
                hasher.update(f.read())
        except Exception:
            pass

    return hasher.hexdigest()


def compute_cohort_fingerprint(cfg: ExperimentConfig, checkpoint_path: Optional[str] = None) -> str:
    """Computes a deterministic SHA-256 fingerprint of configuration and model artifact, EXCLUDING evaluation seed."""
    hasher = hashlib.sha256()
    payload = {
        "scenario_name": cfg.scenario_name,
        "algorithm": cfg.algorithm,
        "training_seed": cfg.training_seed,
        "steps": cfg.steps,
        "yellow_duration": cfg.yellow_duration,
        "all_red_duration": cfg.all_red_duration,
        "min_green": cfg.min_green,
        "max_green": cfg.max_green,
        "fixed_time_green": cfg.fixed_time_green,
        "pressure_metric": cfg.pressure_metric,
        "decision_interval": cfg.decision_interval,
    }
    hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    _hash_scenario_inputs(cfg.scenario_name, hasher)

    target_checkpoint = checkpoint_path or cfg.model_path or resolve_model_path(cfg)
    if target_checkpoint and os.path.isfile(target_checkpoint):
        try:
            with open(target_checkpoint, "rb") as f:
                hasher.update(f.read())
        except Exception:
            pass

    return hasher.hexdigest()


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
        if abs(algo_val) < 1e-9:
            return 0.0
        if higher_is_better:
            return 100.0 if algo_val > 0 else -100.0
        else:
            # Baseline was ideal (0.0), algorithm produced positive degradation (e.g. collisions)
            return -100.0 * algo_val if algo_val > 0 else 0.0

    if higher_is_better:
        return ((algo_val - baseline_val) / abs(baseline_val)) * 100.0
    else:
        return ((baseline_val - algo_val) / abs(baseline_val)) * 100.0


def validate_run_metrics(
    metrics: Dict[str, Any],
    total_demand: int,
    return_flags: bool = False,
) -> Union[Tuple[bool, List[str]], Tuple[bool, List[str], Dict[str, str]]]:
    """Performs rigorous anomaly detection and scientific integrity checks on simulation results.

    Returns:
        Tuple of (is_valid, list_of_anomaly_messages) or (is_valid, list_of_anomaly_messages, status_flags).
    """
    critical_anomalies: List[str] = []
    other_anomalies: List[str] = []

    # Check for NaN / Infinity
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if math.isnan(v):
                critical_anomalies.append(f"Metric '{k}' has NaN value")
            elif math.isinf(v):
                critical_anomalies.append(f"Metric '{k}' has Infinity value")

    # Waiting time checks
    mean_wait = metrics.get("mean_waiting_time", 0.0)
    trip_wait = metrics.get("tripinfo_mean_waiting_time", mean_wait)
    if mean_wait < 0.0 or trip_wait < 0.0:
        critical_anomalies.append(f"Negative waiting time detected: mean={mean_wait}, tripinfo={trip_wait}")

    # Queue length checks
    queue_len = metrics.get("mean_queue_length", 0.0)
    if queue_len < 0.0:
        critical_anomalies.append(f"Negative queue length detected: {queue_len}")

    # Travel time / Time loss checks
    travel_time = metrics.get("mean_travel_time", 0.0)
    if travel_time < 0.0:
        critical_anomalies.append(f"Negative travel time detected: {travel_time}")

    time_loss = metrics.get("total_time_loss", 0.0)
    if time_loss < 0.0:
        critical_anomalies.append(f"Negative total time loss detected: {time_loss}")

    # Completion count & rate checks
    completed = metrics.get("total_completed_vehicles", 0)
    if completed < 0:
        critical_anomalies.append(f"Negative completed vehicles count: {completed}")

    # If full horizon run (total_steps >= 1800), having zero completed vehicles is a critical anomaly
    total_steps = metrics.get("total_steps", 0)
    if total_steps >= 1800 and completed == 0:
        critical_anomalies.append(f"Abnormal traffic lock: 0 completed vehicles in {total_steps}s")

    # Safety checks
    coll_count = metrics.get("collision_count", 0) or 0
    tele_count = metrics.get("teleport_count", 0) or 0
    if coll_count > 0 or tele_count > 0:
        other_anomalies.append(f"{coll_count} collisions and {tele_count} teleports detected")
        safety_status = "INCIDENTS_DETECTED"
    else:
        safety_status = "SAFE"

    # Service completion checks
    if total_demand > 0:
        completion_rate = (completed / total_demand) * 100.0
        if completion_rate > 105.0:  # Allow 5% margin for any synthetic additions
            critical_anomalies.append(f"Completion rate exceeds 100%: {completion_rate:.1f}% ({completed}/{total_demand})")
            service_status = "OVERFLOW"
        elif completion_rate < 50.0:
            other_anomalies.append(f"Severe starvation: Completion rate {completion_rate:.1f}% ({completed}/{total_demand})")
            service_status = "SEVERE_STARVATION"
        elif completion_rate < 95.0:
            other_anomalies.append(f"Low completion: Completion rate {completion_rate:.1f}% ({completed}/{total_demand})")
            service_status = "PARTIAL"
        else:
            service_status = "COMPLETED"
    else:
        critical_anomalies.append(f"Zero traffic demand: scenario has no vehicle trips or flows (total_demand={total_demand})")
        service_status = "ZERO_DEMAND"

    is_valid = len(critical_anomalies) == 0
    all_anomalies = critical_anomalies + other_anomalies
    flags = {
        "execution_status": "SUCCESS" if total_steps > 0 else "CRASH",
        "data_validity": "VALID" if is_valid else "INVALID",
        "safety_status": safety_status,
        "service_status": service_status,
    }

    if return_flags:
        return is_valid, all_anomalies, flags
    return is_valid, all_anomalies


def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Executes a single experiment run under standardized evaluation protocol."""
    scenario_info = inspect_scenario(cfg.scenario_name, data_dir="data")
    total_demand = scenario_info.trips_or_vehicles_count

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

    summary: Dict[str, Any] = {}
    episode_reward: Optional[float] = None
    checkpoint_id: Optional[str] = os.path.basename(model_path) if model_path else None
    env: Optional[TrafficSignalRLEnv] = None
    runner: Optional[SimulationRunner] = None

    effective_decision_interval = resolve_effective_decision_interval(cfg)
    effective_min_green = float(cfg.min_green)
    effective_yellow_duration = float(cfg.yellow_duration)
    effective_all_red_duration = float(cfg.all_red_duration)
    try:
        if algo_key in ("q_learning", "qlearning"):
            if scenario_info.tls_count > 1:
                raise ValueError(
                    f"RL algorithm '{cfg.algorithm}' currently requires a single-intersection scenario, "
                    f"got '{cfg.scenario_name}' with {scenario_info.tls_count} intersections. "
                    f"Multi-agent RL (MARL) is required for multi-intersection coordination."
                )
            # RL Agents evaluated in evaluation mode (zero exploration, zero learning updates)
            ckpt_params = load_checkpoint_env_params(model_path)
            effective_decision_interval = resolve_effective_decision_interval(cfg, ckpt_params=ckpt_params)

            # Safety constraint validation between checkpoint and benchmark config
            effective_min_green = float(ckpt_params.get("min_green", cfg.min_green))
            effective_yellow_duration = float(ckpt_params.get("yellow_duration", cfg.yellow_duration))
            effective_all_red_duration = float(ckpt_params.get("all_red_duration", cfg.all_red_duration))

            mismatches = []
            if "min_green" in ckpt_params and abs(float(ckpt_params["min_green"]) - cfg.min_green) > 1e-4:
                mismatches.append(f"min_green (ckpt={ckpt_params['min_green']}s vs cfg={cfg.min_green}s)")
            if "yellow_duration" in ckpt_params and abs(float(ckpt_params["yellow_duration"]) - cfg.yellow_duration) > 1e-4:
                mismatches.append(f"yellow_duration (ckpt={ckpt_params['yellow_duration']}s vs cfg={cfg.yellow_duration}s)")
            if "all_red_duration" in ckpt_params and abs(float(ckpt_params["all_red_duration"]) - cfg.all_red_duration) > 1e-4:
                mismatches.append(f"all_red_duration (ckpt={ckpt_params['all_red_duration']}s vs cfg={cfg.all_red_duration}s)")

            if mismatches:
                raise ValueError(
                    f"Safety constraint mismatch for '{cfg.algorithm}': checkpoint specifies "
                    f"{', '.join(mismatches)}. Benchmark comparison requires identical safety constraints."
                )

            env_kwargs: Dict[str, Any] = {
                "scenario_name": cfg.scenario_name,
                "decision_interval": effective_decision_interval,
                "min_green": effective_min_green,
                "max_green": cfg.max_green,
                "yellow_duration": effective_yellow_duration,
                "all_red_duration": effective_all_red_duration,
                "max_steps": cfg.steps,
                "seed": cfg.seed,
                "output_dir": run_dir,
                "use_tripinfo": True,
                "gui": False,
            }
            if "max_queue_scale" in ckpt_params:
                env_kwargs["max_queue_scale"] = ckpt_params["max_queue_scale"]
            if "max_time_scale" in ckpt_params:
                env_kwargs["max_time_scale"] = ckpt_params["max_time_scale"]
            if "state_discretizer" in ckpt_params:
                env_kwargs["state_discretizer"] = ckpt_params["state_discretizer"]

            env = TrafficSignalRLEnv(**env_kwargs)

            agent = QLearningController(
                num_actions=env.num_actions,
                discretizer=env.state_discretizer,
                min_green=effective_min_green,
                max_green=cfg.max_green,
                seed=cfg.seed,
                eval_mode=True,
            )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Q-Learning model checkpoint not found: {model_path}")
            agent.load(model_path)
            agent.eval_mode = True
            agent.epsilon = 0.0  # Strict zero exploration

            obs_vec, info = env.reset(seed=cfg.seed)
            valid_actions = info["valid_actions"]
            discrete_state = info["discrete_state"]

            accumulated_reward = 0.0
            step_count = 0

            while True:
                act = agent.select_action_for_state(
                    discrete_state, valid_actions=valid_actions, current_phase=discrete_state[0]
                )

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

        else:
            # Baseline controllers: Fixed-Time, Max-Pressure
            controller = None
            if cfg.algorithm == "Fixed-Time":
                controller = FixedTimeController(default_green_time=cfg.fixed_time_green)
                control_mode = "agent"
            elif cfg.algorithm == "Max-Pressure":
                controller = MaxPressureController(
                    min_green=cfg.min_green,
                    max_green=cfg.max_green,
                    pressure_metric=cfg.pressure_metric,
                )
                control_mode = "agent"
            else:
                raise ValueError(
                    f"Unknown algorithm: {cfg.algorithm}. "
                    "Benchmark evaluation only compares traffic signal control algorithms "
                    "(Fixed-Time, Max-Pressure, Q-Learning)."
                )

            sim_cfg = get_scenario_config(
                dataset_name=cfg.scenario_name,
                gui=False,
                step_length=1.0,
                yellow_duration=effective_yellow_duration,
                all_red_duration=effective_all_red_duration,
                min_green=effective_min_green,
                max_green=cfg.max_green,
                max_steps=cfg.steps,
                seed=cfg.seed,
                output_dir=run_dir,
                control_mode=control_mode,
                use_tripinfo=True,
                decision_interval=effective_decision_interval,
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
            "execution_status": "FAILED",
            "is_valid": False,
            "error_message": err_msg,
            "anomalies": [err_msg],
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

    # Extract metrics
    completed = summary.get("total_completed_vehicles", 0)
    completion_rate = round((completed / total_demand * 100.0) if total_demand > 0 else 0.0, 2)
    throughput_vph = round(completed * (3600.0 / max(1, summary.get("total_steps", 3600))), 1)

    is_valid, anomalies, flags = validate_run_metrics(summary, total_demand=total_demand, return_flags=True)
    data_valid = is_valid and summary.get("is_valid", True)
    if not data_valid:
        overall_status = "INVALID"
    elif flags["safety_status"] == "INCIDENTS_DETECTED":
        overall_status = "SAFETY_INCIDENTS"
    elif flags["service_status"] in ("PARTIAL", "SEVERE_STARVATION"):
        overall_status = "LOW_COMPLETION"
    else:
        overall_status = "SUCCESS"

    fingerprint = compute_run_fingerprint(cfg, model_path)
    cohort_id = compute_cohort_fingerprint(cfg, model_path)

    run_record: Dict[str, Any] = {
        "run_id": run_id,
        "scenario_name": cfg.scenario_name,
        "algorithm": cfg.algorithm,
        "seed": cfg.seed,
        "training_seed": cfg.training_seed,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": model_path if model_path else None,
        "fingerprint": fingerprint,
        "cohort_id": cohort_id,
        "decision_interval": effective_decision_interval,
        "min_green": effective_min_green,
        "yellow_duration": effective_yellow_duration,
        "all_red_duration": effective_all_red_duration,
        "fixed_time_green": cfg.fixed_time_green,
        "pressure_metric": cfg.pressure_metric,
        "steps": cfg.steps,
        "status": overall_status,
        "execution_status": flags["execution_status"],
        "data_validity": flags["data_validity"],
        "safety_status": flags["safety_status"],
        "service_status": flags["service_status"],
        "is_valid": data_valid,
        "total_steps": summary.get("total_steps", 0),
        "total_demand": total_demand,
        "total_completed_vehicles": completed,
        "completion_rate_pct": completion_rate,
        "throughput_vph": throughput_vph,
        "mean_waiting_time": summary.get("mean_waiting_time", 0.0),
        "completed_mean_waiting_time": summary.get("completed_mean_waiting_time", summary.get("mean_waiting_time", 0.0)),
        "system_mean_delay": summary.get("system_mean_delay", summary.get("mean_waiting_time", 0.0)),
        "system_total_delay": summary.get("system_total_delay", summary.get("total_waiting_time", 0.0)),
        "total_pending_delay": summary.get("total_pending_delay", 0.0),
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
        "sumo_version": get_sumo_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "output_dir": run_dir,
    }

    # Save summary JSON in run directory
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    print(
        f"==> Status: {overall_status} | Time: {wall_clock:.2f}s | "
        f"Completed: {completed}/{total_demand} ({completion_rate:.1f}%) | "
        f"Mean Wait: {run_record['tripinfo_mean_waiting_time']:.2f}s | "
        f"Mean Queue: {run_record['mean_queue_length']:.2f} | "
        f"Teleports: {run_record['teleport_count']}"
    )
    if anomalies:
        print(f"    [WARNINGS/ANOMALIES]: {anomalies}")

    return run_record


def get_run_cohort_key(r: Dict[str, Any]) -> str:
    """Returns a unique cohort key distinguishing different configurations / models across seeds.

    Crucially, this key DOES NOT vary with evaluation 'seed', allowing runs with different
    evaluation seeds under the same configuration/model to be aggregated together into a single cohort.
    """
    if r.get("cohort_id"):
        return str(r["cohort_id"])
    if r.get("cohort_key"):
        return str(r["cohort_key"])
    parts = [
        str(r.get("scenario_name") or ""),
        str(r.get("algorithm") or ""),
        str(r.get("checkpoint_id") or "no_ckpt"),
        str(r.get("decision_interval") or ""),
        str(r.get("min_green") or ""),
        str(r.get("yellow_duration") or ""),
        str(r.get("all_red_duration") or ""),
        str(r.get("fixed_time_green") or ""),
        str(r.get("pressure_metric") or ""),
        str(r.get("steps") or ""),
    ]
    cfg = r.get("config")
    if isinstance(cfg, dict):
        parts.append(str(sorted(cfg.items())))
    ctrl_cfg = r.get("controller_config")
    if isinstance(ctrl_cfg, dict):
        parts.append(str(sorted(ctrl_cfg.items())))
    return "|".join(parts)


def aggregate_benchmark_statistics(
    run_records: List[Dict[str, Any]],
    baseline_algo: str = "Fixed-Time",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Computes comprehensive mean, std, min, max, and percentage improvements across runs."""
    # Group by (scenario, algorithm, cohort_key)
    cohort_groups: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    failed_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for r in run_records:
        scen = r.get("scenario_name", "unknown")
        algo = r.get("algorithm", "unknown")
        if not r.get("is_valid", False) or r.get("status") == "FAILED":
            failed_groups.setdefault(scen, {}).setdefault(algo, []).append(r)
            continue
        c_key = get_run_cohort_key(r)
        cohort_groups.setdefault(scen, {}).setdefault(algo, {}).setdefault(c_key, []).append(r)

    metrics_keys = [
        ("tripinfo_mean_waiting_time", False),
        ("mean_queue_length", False),
        ("total_completed_vehicles", True),
        ("completion_rate_pct", True),
        ("throughput_vph", True),
        ("tripinfo_mean_travel_time", False),
        ("tripinfo_total_time_loss", False),
        ("system_mean_delay", False),
        ("system_total_delay", False),
        ("total_pending_delay", False),
        ("pending_insertion_vehicles", False),
        ("running_vehicles_at_end", False),
        ("collision_count", False),
        ("teleport_count", False),
        ("phase_change_count", False),
    ]

    summary_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    run_order = {id(r): idx for idx, r in enumerate(run_records)}

    # Ensure all scenarios present in run_records exist in summary_stats
    all_scenarios = sorted(list(set(list(cohort_groups.keys()) + list(failed_groups.keys()))))
    for scen in all_scenarios:
        summary_stats[scen] = {}

    for scen, algos in cohort_groups.items():
        # 1. Compute stats for each algorithm and its cohorts
        for algo, cohorts in algos.items():
            cohort_stat_list: List[Dict[str, Any]] = []

            for c_key, runs in cohorts.items():
                seed_map: Dict[int, Dict[str, Any]] = {}
                for r in runs:
                    seed_map[r["seed"]] = r
                deduped_runs = list(seed_map.values())
                distinct_seeds = sorted(list(seed_map.keys()))
                latest_run_order = max(run_order.get(id(r), 0) for r in runs)

                cohort_stats: Dict[str, Any] = {
                    "run_count": len(deduped_runs),
                    "seeds": distinct_seeds,
                    "cohort_id": c_key,
                    "latest_run_order": latest_run_order,
                    "fingerprint": deduped_runs[-1].get("fingerprint"),
                    "checkpoint_id": deduped_runs[-1].get("checkpoint_id"),
                    "total_steps": int(deduped_runs[-1].get("total_steps", 3600)),
                    "status": "SUCCESS",
                    "is_valid": True,
                }
                for key, _ in metrics_keys:
                    vals = [float(r.get(key, 0.0)) for r in deduped_runs]
                    if vals:
                        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        entry = {
                            "mean": round(float(np.mean(vals)), 2),
                            "std": round(std_val, 2),
                            "min": round(float(np.min(vals)), 2),
                            "max": round(float(np.max(vals)), 2),
                        }
                        if key in ("collision_count", "teleport_count"):
                            entry["sum"] = int(np.sum(vals))
                        cohort_stats[key] = entry
                    else:
                        entry = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                        if key in ("collision_count", "teleport_count"):
                            entry["sum"] = 0
                        cohort_stats[key] = entry

                safety_counts: Dict[str, int] = {}
                service_counts: Dict[str, int] = {}
                for r in deduped_runs:
                    s_stat = r.get("safety_status", "UNKNOWN")
                    safety_counts[s_stat] = safety_counts.get(s_stat, 0) + 1
                    srv_stat = r.get("service_status", "UNKNOWN")
                    service_counts[srv_stat] = service_counts.get(srv_stat, 0) + 1
                cohort_stats["safety_status_counts"] = safety_counts
                cohort_stats["service_status_counts"] = service_counts

                # Optional RL reward
                rewards = [r["episode_reward"] for r in deduped_runs if r.get("episode_reward") is not None]
                if rewards:
                    std_rew = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
                    cohort_stats["episode_reward"] = {
                        "mean": round(float(np.mean(rewards)), 2),
                        "std": round(std_rew, 2),
                        "min": round(float(np.min(rewards)), 2),
                        "max": round(float(np.max(rewards)), 2),
                    }

                # Attach failed run details if any seed failed
                if algo in failed_groups.get(scen, {}):
                    f_runs = failed_groups[scen][algo]
                    cohort_stats["failed_count"] = len(f_runs)
                    err_msgs = [fr.get("error_message") or (fr.get("anomalies")[0] if fr.get("anomalies") else "Simulation failed") for fr in f_runs]
                    cohort_stats["error_messages"] = list(dict.fromkeys(err_msgs))

                cohort_stat_list.append(cohort_stats)

            # The primary cohort represents the algorithm: picked by most recent run in history
            latest_cohort = max(cohort_stat_list, key=lambda cs: cs["latest_run_order"])
            summary_stats[scen][algo] = latest_cohort

            # If multiple cohorts exist for this algorithm, also expose each cohort explicitly
            if len(cohort_stat_list) > 1:
                latest_cohort["has_multiple_cohorts"] = True
                latest_cohort["total_cohorts"] = len(cohort_stat_list)
                cohort_stat_list.sort(key=lambda cs: cs["latest_run_order"])
                for cs in cohort_stat_list:
                    c_id = cs["cohort_id"]
                    fp = cs.get("fingerprint")
                    label = f"{algo} (fp: {fp[:8]})" if fp else f"{algo} (cohort: {c_id[:8]})"
                    summary_stats[scen][label] = cs

        # 2. Compute improvements relative to baseline
        baseline_stats = summary_stats[scen].get(baseline_algo)
        if baseline_stats and baseline_stats.get("is_valid", False):
            b_steps = baseline_stats.get("total_steps")
            for algo_key, stats in summary_stats[scen].items():
                if algo_key == baseline_algo or not stats.get("is_valid", False):
                    continue
                a_steps = stats.get("total_steps")
                if b_steps and a_steps and b_steps != a_steps:
                    stats["protocol_incompatible"] = True
                    stats["improvements_vs_baseline"] = {}
                    continue
                improvements: Dict[str, float] = {}
                for key, higher_is_better in metrics_keys:
                    if key in baseline_stats and key in stats:
                        b_mean = baseline_stats[key]["mean"]
                        a_mean = stats[key]["mean"]
                        imp = compute_improvement(b_mean, a_mean, higher_is_better=higher_is_better)
                        improvements[f"{key}_improvement_pct"] = round(imp, 2)
                stats["improvements_vs_baseline"] = improvements

    # Record algorithms where ALL runs failed
    for scen, f_algos in failed_groups.items():
        summary_stats.setdefault(scen, {})
        for algo, f_runs in f_algos.items():
            if algo not in summary_stats[scen]:
                err_msgs = [fr.get("error_message") or (fr.get("anomalies")[0] if fr.get("anomalies") else "Simulation failed") for fr in f_runs]
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

    # Build set of already executed fingerprints
    executed_fingerprints = {
        r.get("fingerprint")
        for r in all_records
        if r.get("is_valid", False) and r.get("status") == "SUCCESS" and r.get("fingerprint")
    }

    total_tasks = len(scenarios) * len(algorithms) * len(seeds)
    task_idx = 0

    print(f"\n================================================================================")
    print(f"STARTING COMPREHENSIVE BENCHMARK EVALUATION (PHASE 6)")
    print(f"Scenarios: {scenarios}")
    print(f"Algorithms: {algorithms}")
    print(f"Evaluation Seeds: {seeds} (Total Tasks: {total_tasks})")
    print(f"Config: all_red={default_all_red}s, yellow={default_yellow}s, min_green={default_min_green}s, max_green={default_max_green}s")
    print(f"================================================================================")

    for scen in scenarios:
        for algo in algorithms:
            for s in seeds:
                task_idx += 1
                cfg = ExperimentConfig(
                    scenario_name=scen,
                    algorithm=algo,
                    seed=s,
                    training_seed=42,
                    output_dir=output_dir,
                    yellow_duration=default_yellow,
                    all_red_duration=default_all_red,
                    min_green=default_min_green,
                    max_green=default_max_green,
                    steps=default_steps,
                )
                model_p = resolve_model_path(cfg)
                cfg.model_path = model_p
                fp = compute_run_fingerprint(cfg, model_p)
                if resume and fp in executed_fingerprints:
                    print(f"[{task_idx}/{total_tasks}] Skipping already completed: {scen} | {algo} | Seed {s} (fp: {fp[:8]})")
                    continue

                print(f"\n[{task_idx}/{total_tasks}] Executing: {scen} | {algo} | Seed {s}")
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
    parser = argparse.ArgumentParser(description="Phase 6 Experiment Runner")
    parser.add_argument("--scenario", type=str, default="cologne1", help="Single scenario name (default: cologne1)")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenarios (e.g. 'cologne1' or 'cologne1,ingolstadt1')")
    parser.add_argument("--algorithm", type=str, default="Fixed-Time", help="Single algorithm name or alias (e.g. ft, mp, ql, Fixed-Time)")
    parser.add_argument("--algorithms", type=str, default=None, help="Comma-separated algorithms/aliases (e.g. 'ft,mp,ql' or 'Fixed-Time,Max-Pressure,Q-Learning')")
    parser.add_argument("--seed", type=int, default=42, help="Single seed (default: 42)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (e.g. '42' or '42,101,2024')")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml", help="Path to YAML configuration file")
    parser.add_argument("--run-all", action="store_true", help="Run full benchmark matrix (all scenarios, algorithms, seeds)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh without loading existing benchmark_runs.json")
    args = parser.parse_args()

    # Determine execution mode
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
        )
        run_single_experiment(cfg)

