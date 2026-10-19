"""Automated Experiment Runner for Traffic Signal Control Benchmarks.

Executes, aggregates, and benchmarks traffic signal control algorithms
(Fixed-Time, Max-Pressure, Q-Learning) in a clean, straightforward workflow.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.config import DEFAULT_SAFETY_CONFIG, get_scenario_config, load_yaml_config
from src.controllers import algorithm_key, make_controller, resolve_q_learning_checkpoint
from src.controllers.q_learning import QLearningController
from src.runner import SimulationRunner
from src.evaluation.evaluate_rl import checkpoint_env_params, create_evaluation_session, run_greedy_eval_loop
from src.metrics import BENCHMARK_METRICS, METRIC_SCHEMA_VERSION, completion_statistics, service_status


def normalize_algo_name(name: str) -> str:
    """Normalizes algorithm names including short aliases like ft, mp, ql."""
    display_names = {"fixed-time": "Fixed-Time", "max-pressure": "Max-Pressure", "q-learning": "Q-Learning"}
    return display_names.get(algorithm_key(name), name.strip())


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
    timing_overrides: Dict[str, Optional[float]] = field(default_factory=dict)
    gui: bool = False
    gui_delay: float = 0.0
    fixed_time_green: float = 30.0
    pressure_metric: str = "queue"
    save_run_details: bool = False
    step_length: float = 1.0
    use_tripinfo: bool = True

    def __post_init__(self):
        self.algorithm = normalize_algo_name(self.algorithm)


def load_experiments_yaml(config_path: str = "configs/experiments.yaml") -> Dict[str, Any]:
    """Loads benchmark parameters from experiments YAML file if present."""
    return load_yaml_config(config_path)


def effective_timing(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Only explicitly supplied overrides may replace checkpoint timing."""
    is_rl = algorithm_key(cfg.algorithm) == "q-learning"
    timing = {**DEFAULT_SAFETY_CONFIG, "decision_interval": 5.0 if is_rl else 1.0}
    model = resolve_model_path(cfg)
    if is_rl and model and Path(model).is_file():
        params = checkpoint_env_params(QLearningController.from_checkpoint(model).get_config())
        timing.update({key: params[key] for key in timing})
    for key, value in cfg.timing_overrides.items():
        if key not in timing or (value is None and key != "max_green"):
            raise ValueError(f"Invalid timing override: {key}={value}")
        timing[key] = value
    return {key: float(value) if value is not None else None for key, value in timing.items()}


def resolve_effective_decision_interval(cfg: ExperimentConfig, ckpt_params=None) -> float:
    if ckpt_params and "decision_interval" not in cfg.timing_overrides:
        return float(ckpt_params.get("decision_interval", effective_timing(cfg)["decision_interval"]))
    return float(effective_timing(cfg)["decision_interval"])


def simulation_options(simulation: Dict[str, Any]) -> Dict[str, Any]:
    """Split YAML timing overrides from ordinary simulation options."""
    keys = (*DEFAULT_SAFETY_CONFIG, "decision_interval")
    return {"timing_overrides": {k: simulation[k] for k in keys if k in simulation},
            **{k: simulation[k] for k in ("step_length", "use_tripinfo", "steps") if k in simulation}}


def resolve_model_path(cfg: ExperimentConfig) -> Optional[str]:
    """Resolves model checkpoint path for RL controllers if not explicitly set."""
    if cfg.model_path:
        return cfg.model_path
    algo_key = algorithm_key(cfg.algorithm)
    if algo_key == "q-learning":
        return resolve_q_learning_checkpoint(cfg.scenario_name)
    return None


def _experiment_identity(cfg: ExperimentConfig, checkpoint_path=None, root=None, include_seed=True) -> str:
    base = Path(root or ".")
    params = asdict(cfg)
    for key in ("output_dir", "model_path"):
        params.pop(key, None)
    params["algorithm"] = algorithm_key(cfg.algorithm)
    params["metric_schema_version"] = METRIC_SCHEMA_VERSION
    try:
        params["timing_overrides"] = effective_timing(cfg)
    except (ValueError, TypeError, KeyError, SyntaxError, AttributeError, OverflowError):
        pass  # Invalid checkpoints still have a stable content fingerprint.

    if not include_seed:
        params.pop("seed", None)
    model = Path(checkpoint_path or resolve_model_path(cfg) or "__no_model__")
    if not model.is_absolute():
        model = base / model
    payload = {
        "config": params,
        "checkpoint": hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else model.as_posix(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compute_run_fingerprint(cfg: ExperimentConfig, checkpoint_path=None, root=None) -> str:
    return _experiment_identity(cfg, checkpoint_path, root)


def compute_cohort_fingerprint(cfg: ExperimentConfig, checkpoint_path=None, root=None) -> str:
    return _experiment_identity(cfg, checkpoint_path, root, include_seed=False)


def _write_json(path, data) -> None:
    """Replace the result only after a complete JSON write succeeds."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def compute_improvement(baseline_val: float, algo_val: float, higher_is_better: bool = False) -> Optional[float]:
    """Calculates relative improvement percentage against a baseline."""
    if not math.isfinite(baseline_val) or not math.isfinite(algo_val):
        return None
    if abs(baseline_val) < 1e-9:
        return 0.0 if abs(algo_val) < 1e-9 else None
    sign = 1 if higher_is_better else -1
    return sign * (algo_val - baseline_val) / abs(baseline_val) * 100.0


def run_single_experiment(cfg: ExperimentConfig, *, include_summary: bool = False):
    """Run simulation or evaluation; optionally return the full metrics summary."""
    start_wall_clock = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.scenario_name}_{algorithm_key(cfg.algorithm)}_s{cfg.seed}_{timestamp}"
    run_dir = os.path.join(cfg.output_dir, "runs", run_id) if cfg.save_run_details else None
    identity = {"run_id": run_id, "scenario_name": cfg.scenario_name, "algorithm": cfg.algorithm, "seed": cfg.seed}
    summary, episode_reward = {}, None
    model_path = resolve_model_path(cfg)
    try:
        identity.update(run_fingerprint=compute_run_fingerprint(cfg), cohort_fingerprint=compute_cohort_fingerprint(cfg))
        timing = effective_timing(cfg)
        options = dict(max_steps=cfg.steps, seed=cfg.seed, output_dir=run_dir,
                       gui=cfg.gui, gui_delay=cfg.gui_delay, step_length=cfg.step_length,
                       use_tripinfo=cfg.use_tripinfo, save_run_details=cfg.save_run_details)
        if not include_summary and not cfg.use_tripinfo:
            raise ValueError("Benchmark evaluation requires TripInfo")
        if algorithm_key(cfg.algorithm) == "q-learning":
            agent, env = create_evaluation_session(cfg.scenario_name, model_path,
                                                 timing_overrides=cfg.timing_overrides, **options)
            with env:
                episode_reward, summary = run_greedy_eval_loop(agent, env, seed=cfg.seed)
        else:
            controller, mode = make_controller(cfg.algorithm, dataset_name=cfg.scenario_name,
                                               fixed_time_green=cfg.fixed_time_green, pressure_metric=cfg.pressure_metric,
                                               min_green=timing["min_green"], max_green=timing["max_green"])
            sim_cfg = get_scenario_config(cfg.scenario_name, control_mode=mode, **timing, **options)
            summary = SimulationRunner(sim_cfg, controller=controller).run()
        if not summary.get("is_valid", True):
            raise ValueError(f"Invalid simulation measurements: {summary.get('measurement_error_messages', [])}")
    except Exception as error:
        record = {**identity, "status": "FAILED", "is_valid": False, "error_message": str(error),
                  "wall_clock_time_seconds": round(time.time() - start_wall_clock, 2)}
        return (record, summary) if include_summary else record
    wall_clock = time.time() - start_wall_clock

    # Extract key metrics cleanly
    completed, scheduled_demand, completion_fraction = completion_statistics(summary)
    completion_rate = round(completion_fraction * 100.0, 2)
    duration = summary.get("observation_window_duration") or (summary.get("total_steps", 0) * summary.get("step_length", 1.0))
    throughput_vph = round(completed * 3600.0 / duration, 1) if duration > 0 else 0.0

    col_count = summary.get("collision_count", 0) or 0
    tele_count = summary.get("teleport_count", 0) or 0
    safety_status = "INCIDENTS_DETECTED" if (col_count > 0 or tele_count > 0) else "SAFE"
    run_service_status = "NO_DEMAND" if scheduled_demand == 0 else service_status(completion_fraction)

    mean_wait = summary.get("mean_waiting_time", 0.0)
    trip_wait = summary.get("tripinfo_mean_waiting_time", mean_wait)

    run_record: Dict[str, Any] = {
        **identity,
        "training_seed": cfg.training_seed,
        "checkpoint_path": model_path,
        "status": "SUCCESS",
        "is_valid": True,
        "total_steps": summary.get("total_steps", 0),
        "observation_window_begin": summary.get("observation_window_begin"),
        "observation_window_end": summary.get("observation_window_end"),
        "observation_window_duration": summary.get("observation_window_duration"),
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
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "phase_durations_per_tls": summary.get("phase_durations_per_tls", {}),
        "reward_config": summary.get("reward_config"),
        "reward_config_source": summary.get("reward_config_source"),
        "effective_config": {"steps": cfg.steps, **{key: summary.get(key, value) for key, value in {**timing, "step_length": cfg.step_length}.items()}},
        "controlled_tls_ids": summary.get("controlled_tls_ids", []),
        "safety_status": safety_status,
        "service_status": run_service_status,
        "wall_clock_time_seconds": round(wall_clock, 2),
        "output_dir": run_dir,
    }

    # Save summary JSON in run directory if requested
    if cfg.save_run_details and run_dir:
        summary_path = os.path.join(run_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)

    wait_label = f"{trip_wait:.2f}s" if trip_wait is not None else "N/A"
    print(
        f"==> Completed in {wall_clock:.1f}s | "
        f"Vehicles: {completed}/{scheduled_demand} ({completion_rate:.1f}%) | "
        f"Wait: {wait_label} | Queue: {run_record['mean_queue_length']:.2f} | "
        f"Collisions: {col_count}"
    )
    return (run_record, summary) if include_summary else run_record


def aggregate_benchmark_statistics(
    run_records: List[Dict[str, Any]],
    baseline_algo: str = "Fixed-Time",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Computes mean, std, and improvement percentages grouped by scenario and algorithm."""
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    failed_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    baseline_algo = normalize_algo_name(baseline_algo)
    unique = {}
    cohorts = {}
    for record in run_records:
        norm_algo = normalize_algo_name(record.get("algorithm", ""))
        key = (record.get("scenario_name"), norm_algo)
        cohort = record.get("cohort_fingerprint")
        if key in cohorts and cohorts[key] != cohort:
            raise ValueError("Cannot aggregate different experiment cohorts")
        cohorts[key] = cohort
        unique[(*key, record.get("seed"))] = {**record, "algorithm": norm_algo}

    for r in unique.values():
        scen = r.get("scenario_name", "unknown")
        algo = r.get("algorithm", "unknown")
        if not r.get("is_valid", False) or r.get("status") == "FAILED":
            failed_groups.setdefault(scen, {}).setdefault(algo, []).append(r)
        else:
            groups.setdefault(scen, {}).setdefault(algo, []).append(r)

    summary_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scen in sorted(groups.keys() | failed_groups.keys()):
        summary_stats[scen] = {}

    for scen, algos in groups.items():
        for algo, runs in algos.items():
            distinct_seeds = sorted({r["seed"] for r in runs if "seed" in r})
            algo_stats: Dict[str, Any] = {
                "effective_config": runs[0].get("effective_config"),
                "reward_config": runs[0].get("reward_config"),
                "controlled_tls_ids": runs[0].get("controlled_tls_ids", []),
                "run_count": len(runs),
                "seeds": distinct_seeds,
                "status": "SUCCESS",
                "is_valid": True,
            }

            for key, _ in (*BENCHMARK_METRICS, ("episode_reward", True)):
                vals = [float(r[key]) for r in runs if r.get(key) is not None and math.isfinite(float(r[key]))]
                entry = {
                    "mean": round(float(np.mean(vals)), 2) if vals else None,
                    "std": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else (0.0 if vals else None),
                    "min": round(min(vals), 2) if vals else None,
                    "max": round(max(vals), 2) if vals else None,
                    "available": bool(vals),
                }
                if key in ("collision_count", "teleport_count"):
                    entry["sum"] = int(sum(vals))
                algo_stats[key] = entry

            col_sum = algo_stats.get("collision_count", {}).get("sum", 0)
            tel_sum = algo_stats.get("teleport_count", {}).get("sum", 0)
            algo_stats["safety_status"] = ("UNKNOWN" if not algo_stats["collision_count"]["available"] or not algo_stats["teleport_count"]["available"] else ("INCIDENTS_DETECTED" if col_sum or tel_sum else "SAFE"))
            cr_mean = algo_stats.get("completion_rate_pct", {}).get("mean", 100.0)
            algo_stats["service_status"] = ("NO_DEMAND" if all(run.get("total_demand") == 0 for run in runs) else service_status(cr_mean / 100.0 if cr_mean is not None else None))

            summary_stats[scen][algo] = algo_stats

        # Compute improvements relative to baseline algorithm
        baseline_stats = summary_stats[scen].get(baseline_algo)
        if baseline_stats and baseline_stats.get("is_valid", False):
            for algo, stats in summary_stats[scen].items():
                if algo == baseline_algo or not stats.get("is_valid", False):
                    continue
                b_tls = set(baseline_stats.get("controlled_tls_ids") or [])
                a_tls = set(stats.get("controlled_tls_ids") or [])
                if b_tls and a_tls and b_tls != a_tls:
                    stats["comparison_unavailable_reason"] = (
                        f"Controlled TLS mismatch in '{scen}': baseline '{baseline_algo}' controls {sorted(b_tls)}, "
                        f"but '{algo}' controls {sorted(a_tls)}. Cannot compare runs with differing control scopes."
                    )
                    continue
                baseline_runs = {run.get("seed"): run for run in algos[baseline_algo]}
                candidate_runs = {run.get("seed"): run for run in algos[algo]}
                if baseline_runs.keys() != candidate_runs.keys() or None in baseline_runs:
                    raise ValueError(f"Cannot compare '{algo}' with '{baseline_algo}' in '{scen}': seed sets differ or are missing")
                verified = True
                window_keys = ("total_steps", "observation_window_begin", "observation_window_end", "observation_window_duration")
                for seed, baseline_run in baseline_runs.items():
                    candidate_run = candidate_runs[seed]
                    base_config = baseline_run.get("effective_config") or {}
                    candidate_config = candidate_run.get("effective_config") or {}
                    # Controller decision intervals may intentionally differ by algorithm.
                    config_keys = ("steps", "step_length", *DEFAULT_SAFETY_CONFIG)
                    base_conditions = {key: base_config.get(key) for key in config_keys}
                    candidate_conditions = {key: candidate_config.get(key) for key in config_keys}
                    for key in (*window_keys, "metric_schema_version"):
                        base_conditions[key] = baseline_run.get(key)
                        candidate_conditions[key] = candidate_run.get(key)
                    if base_conditions != candidate_conditions:
                        raise ValueError(f"Cannot compare '{algo}' with '{baseline_algo}' in '{scen}', seed {seed}: simulation conditions differ")
                    if (any(baseline_run.get(key) is None for key in window_keys)
                            or "steps" not in base_config or base_config.get("step_length") is None):
                        verified = False
                if not verified:
                    stats["comparison_unavailable_reason"] = "Missing simulation window or step configuration"
                    continue
                improvements: Dict[str, float] = {}
                for key, higher_is_better in BENCHMARK_METRICS:
                    if key in baseline_stats and key in stats:
                        b_mean = baseline_stats[key].get("mean")
                        a_mean = stats[key].get("mean")
                        if b_mean is not None and a_mean is not None:
                            imp = compute_improvement(b_mean, a_mean, higher_is_better=higher_is_better)
                            if imp is not None:
                                improvements[f"{key}_improvement_pct"] = round(imp, 2)
                stats["improvements_vs_baseline"] = improvements

    # Preserve partial failure information instead of hiding failed seeds.
    for scen, algorithms in failed_groups.items():
        for algo, failures in algorithms.items():
            stats = summary_stats[scen].setdefault(algo, {"run_count": 0, "seeds": [], "is_valid": False})
            stats.update(
                successful_count=stats["run_count"], failed_count=len(failures),
                failed_seeds=sorted(record["seed"] for record in failures),
                status="PARTIAL" if stats["run_count"] else "FAILED",
                error_messages=list(dict.fromkeys(record.get("error_message", "Simulation failed") for record in failures)),
            )
    for algorithms in summary_stats.values():
        for stats in algorithms.values():
            stats.setdefault("successful_count", stats["run_count"])
            stats.setdefault("failed_count", 0)

    return summary_stats


def run_full_benchmark_matrix(
    scenarios: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    config_file: str = "configs/experiments.yaml",
    output_dir: str = "results/evaluation",
    resume: bool = True,
    save_run_details: bool = False,
    generate_report: bool = True,
    generate_plots: bool = False,
    figures_dir: str = "results/figures",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs full evaluation benchmark across all scenarios, algorithms, and seeds."""
    yaml_cfg = load_experiments_yaml(config_file)
    sim_yaml = yaml_cfg.get("simulation", {})

    if scenarios is None:
        scenarios = yaml_cfg.get("scenarios", ["cologne1", "ingolstadt1"])
    if algorithms is None:
        algorithms = yaml_cfg.get("algorithms", ["Fixed-Time", "Max-Pressure", "Q-Learning"])
    if seeds is None:
        seeds = yaml_cfg.get("seeds", [sim_yaml.get("seed", 42)])

    options = simulation_options(sim_yaml)

    os.makedirs(output_dir, exist_ok=True)
    master_records_file = os.path.join(output_dir, "benchmark_runs.json")
    all_records: List[Dict[str, Any]] = []

    if resume and os.path.isfile(master_records_file):
        all_records = json.loads(Path(master_records_file).read_text(encoding="utf-8"))
        if not isinstance(all_records, list):
            raise ValueError("Benchmark records must be a JSON list")
    executed_keys = {record.get("run_fingerprint") for record in all_records
                     if record.get("status") == "SUCCESS" and record.get("is_valid")}
    requested_keys = set()

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
                cfg = ExperimentConfig(
                    scenario_name=scen,
                    algorithm=algo,
                    seed=s,
                    output_dir=output_dir,
                    **options,
                    save_run_details=save_run_details,
                )
                try:
                    fingerprint = compute_run_fingerprint(cfg)
                    requested_keys.add(fingerprint)
                    if resume and fingerprint in executed_keys:
                        print(f"[{task_idx}/{total_tasks}] Reusing matching run: {scen} | {algo} | {s}")
                        continue
                    rec = run_single_experiment(cfg)
                    rec.update(run_fingerprint=fingerprint, cohort_fingerprint=compute_cohort_fingerprint(cfg))
                except Exception as task_err:
                    fingerprint = getattr(cfg, "run_id", f"{scen}_{algo}_{s}_failed")
                    requested_keys.add(fingerprint)
                    rec = {
                        "scenario_name": scen,
                        "algorithm": algo,
                        "seed": s,
                        "status": "FAILED",
                        "is_valid": False,
                        "error_message": str(task_err),
                        "run_fingerprint": fingerprint,
                    }
                all_records.append(rec)
                _write_json(master_records_file, all_records)

    selected = [record for record in all_records if record.get("run_fingerprint") in requested_keys]
    summary_stats = aggregate_benchmark_statistics(selected, baseline_algo="Fixed-Time")
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    import pandas as pd
    csv_file = os.path.join(output_dir, "benchmark_runs.csv")
    pd.DataFrame(selected).to_csv(csv_file, index=False)

    if generate_report:
        from src.evaluation.compare_algorithms import generate_comparison_report
        report_md = os.path.join(output_dir, "benchmark_comparison.md")
        generate_comparison_report(
            eval_dir=output_dir,
            phase6_summary=summary_file,
            output_report=report_md,
        )

    if generate_plots:
        from src.evaluation.plot_comparison import generate_all_phase6_charts
        generate_all_phase6_charts(
            output_dir=figures_dir,
            summary_file=summary_file,
            runs_file=master_records_file,
            selected_records=selected,
        )

    print(f"\n================================================================================")
    print(f"BENCHMARK COMPLETED: Results saved in '{output_dir}'")
    print(f"================================================================================")
    return selected, summary_stats


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    parser.add_argument("--report", action=argparse.BooleanOptionalAction, default=True, help="Auto-generate Markdown comparison report")
    parser.add_argument("--plots", action="store_true", default=False, help="Auto-generate publication-quality charts")
    args = parser.parse_args(argv)

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
        current_records, summary_stats = run_full_benchmark_matrix(
            scenarios=scenarios_list,
            algorithms=algorithms_list,
            seeds=seeds_list,
            config_file=args.config,
            resume=not args.fresh,
            save_run_details=args.save_details,
            generate_report=args.report,
            generate_plots=args.plots,
        )
        has_failure = any(r.get("status") == "FAILED" or not r.get("is_valid", True) for r in current_records)
        return 1 if has_failure else 0
    else:
        norm_algo = normalize_algo_name(args.algorithm)
        yaml_cfg = load_experiments_yaml(args.config)
        sim_yaml = yaml_cfg.get("simulation", {})
        cfg = ExperimentConfig(
            scenario_name=args.scenario,
            algorithm=norm_algo,
            seed=args.seed,
            **simulation_options(sim_yaml),
            save_run_details=args.save_details,
        )
        record = run_single_experiment(cfg)
        if record.get("status") == "FAILED" or not record.get("is_valid", True):
            err_msg = record.get("error_message") or "Unknown error"
            import sys
            sys.stderr.write(f"Experiment failed: {err_msg}\n")
            return 1
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
