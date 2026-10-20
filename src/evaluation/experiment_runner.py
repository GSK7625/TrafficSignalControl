"""Automated Experiment Runner for Traffic Signal Control Benchmarks.

Executes, aggregates, and benchmarks traffic signal control algorithms
(Fixed-Time, Max-Pressure, Q-Learning) in a clean, straightforward workflow.
"""

from dataclasses import asdict, dataclass, field
import csv
from datetime import datetime
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.config import DEFAULT_SAFETY_CONFIG, get_scenario_config, load_yaml_config, resolve_config_path
from src.controllers import algorithm_key, make_controller, resolve_q_learning_checkpoint
from src.controllers.q_learning import QLearningController
from src.runner import SimulationRunner
from src.evaluation.evaluate_rl import create_evaluation_session, run_greedy_eval_loop
from src.metrics import BENCHMARK_METRICS, METRIC_SCHEMA_VERSION, completion_statistics, service_status
from src.io_utils import write_json_atomic as _write_json


def normalize_algo_name(name: str) -> str:
    """Normalizes algorithm names including short aliases like ft, mp, ql."""
    display = {"fixed-time": "Fixed-Time", "max-pressure": "Max-Pressure", "q-learning": "Q-Learning"}
    return display.get(algorithm_key(name), name.strip())


@dataclass
class ExperimentConfig:
    """Configuration for a single benchmark evaluation run."""
    scenario_name: str
    algorithm: str
    seed: int = 42
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


def load_experiments_yaml(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads benchmark parameters from YAML config."""
    config_path = resolve_config_path(config_path)
    raw = load_yaml_config(config_path)
    if "benchmark" in raw:
        bench = dict(raw["benchmark"])
        bench["simulation"] = {
            **raw.get("simulation", {}), **raw.get("safety", {}),
            **({"steps": bench["steps"]} if "steps" in bench else {}),
            **bench.get("simulation", {}),
        }
        return bench
    return raw


def resolve_model_path(cfg: ExperimentConfig) -> Optional[str]:
    """Resolves model checkpoint path for RL controllers if not explicitly set."""
    if cfg.model_path:
        return cfg.model_path
    if algorithm_key(cfg.algorithm) == "q-learning":
        return resolve_q_learning_checkpoint(cfg.scenario_name)
    return None


def effective_timing(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Resolves timing constraints and explicit overrides."""
    is_rl = algorithm_key(cfg.algorithm) == "q-learning"
    timing = {**DEFAULT_SAFETY_CONFIG, "decision_interval": 5.0 if is_rl else 1.0, "max_red": None}
    model = resolve_model_path(cfg)
    if is_rl and model and Path(model).is_file():
        params = QLearningController.from_checkpoint(model).get_config()
        timing.update({k: params[k] for k in timing if k in params})
    for k, v in cfg.timing_overrides.items():
        if k not in timing or (v is None and k not in ("max_green", "max_red")):
            raise ValueError(f"Invalid timing override: {k}={v}")
        timing[k] = v
    return {k: float(v) if v is not None else None for k, v in timing.items()}


def simulation_options(simulation: Dict[str, Any]) -> Dict[str, Any]:
    """Splits YAML timing overrides from simulation options."""
    keys = (*DEFAULT_SAFETY_CONFIG, "decision_interval", "max_red")
    return {
        "timing_overrides": {k: simulation[k] for k in keys if k in simulation},
        **{k: simulation[k] for k in ("step_length", "use_tripinfo", "steps") if k in simulation},
    }


def _experiment_identity(cfg: ExperimentConfig, checkpoint_path=None, root=None, include_seed=True) -> str:
    base = Path(root or ".")
    params = asdict(cfg)
    for k in ("output_dir", "model_path"):
        params.pop(k, None)
    params["algorithm"] = algorithm_key(cfg.algorithm)
    params["metric_schema_version"] = METRIC_SCHEMA_VERSION
    try:
        params["timing_overrides"] = effective_timing(cfg)
    except Exception:
        pass

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


def compute_improvement(baseline_val: float, algo_val: float, higher_is_better: bool = False) -> Optional[float]:
    """Calculates relative improvement percentage against a baseline."""
    if not math.isfinite(baseline_val) or not math.isfinite(algo_val):
        return None
    if abs(baseline_val) < 1e-9:
        return 0.0 if abs(algo_val) < 1e-9 else None
    sign = 1 if higher_is_better else -1
    return sign * (algo_val - baseline_val) / abs(baseline_val) * 100.0


def run_single_experiment(cfg: ExperimentConfig, *, include_summary: bool = False):
    """Executes a single simulation or RL evaluation run."""
    start_time = time.time()
    algo_key = algorithm_key(cfg.algorithm)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.scenario_name}_{algo_key}_s{cfg.seed}_{timestamp}"
    run_dir = os.path.join(cfg.output_dir, "runs", run_id) if cfg.save_run_details else None
    identity = {"run_id": run_id, "scenario_name": cfg.scenario_name, "algorithm": cfg.algorithm, "seed": cfg.seed}
    summary, episode_reward, model_path = {}, None, None

    try:
        if not include_summary and not cfg.use_tripinfo:
            raise ValueError("Benchmark evaluation requires TripInfo")
        model_path = resolve_model_path(cfg)
        identity.update(run_fingerprint=compute_run_fingerprint(cfg), cohort_fingerprint=compute_cohort_fingerprint(cfg))
        timing = effective_timing(cfg)
        options = dict(max_steps=cfg.steps, seed=cfg.seed, output_dir=run_dir,
                       gui=cfg.gui, gui_delay=cfg.gui_delay, step_length=cfg.step_length,
                       use_tripinfo=cfg.use_tripinfo, save_run_details=cfg.save_run_details)

        if algo_key == "q-learning":
            agent, env = create_evaluation_session(cfg.scenario_name, model_path,
                                                  timing_overrides=cfg.timing_overrides, **options)
            with env:
                episode_reward, summary = run_greedy_eval_loop(agent, env, seed=cfg.seed)
        else:
            ctrl, mode = make_controller(cfg.algorithm, dataset_name=cfg.scenario_name,
                                         fixed_time_green=cfg.fixed_time_green, pressure_metric=cfg.pressure_metric,
                                         min_green=timing["min_green"], max_green=timing["max_green"])
            sim_cfg = get_scenario_config(cfg.scenario_name, control_mode=mode, **timing, **options)
            summary = SimulationRunner(sim_cfg, controller=ctrl).run()

        if not summary.get("is_valid", True):
            raise ValueError(f"Invalid simulation measurements: {summary.get('measurement_error_messages', [])}")
    except Exception as err:
        record = {**identity, "status": "FAILED", "is_valid": False, "error_message": str(err),
                  "wall_clock_time_seconds": round(time.time() - start_time, 2)}
        return (record, summary) if include_summary else record

    wall_clock = time.time() - start_time
    completed, demand, fraction = completion_statistics(summary)
    duration = summary.get("observation_window_duration") or (summary.get("total_steps", 0) * summary.get("step_length", 1.0))
    mean_wait = summary.get("mean_waiting_time", 0.0)
    trip_wait = summary.get("tripinfo_mean_waiting_time", mean_wait)
    col_count = summary.get("collision_count", 0) or 0
    tele_count = summary.get("teleport_count", 0) or 0

    record: Dict[str, Any] = {
        **identity,
        "checkpoint_path": model_path,
        "status": "SUCCESS",
        "is_valid": True,
        "total_demand": demand,
        "total_completed_vehicles": completed,
        "completion_rate_pct": round(fraction * 100.0, 2),
        "throughput_vph": round(completed * 3600.0 / duration, 1) if duration > 0 else 0.0,
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
        "safety_status": "INCIDENTS_DETECTED" if (col_count > 0 or tele_count > 0) else "SAFE",
        "service_status": "NO_DEMAND" if demand == 0 else service_status(fraction),
        "episode_reward": episode_reward,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "wall_clock_time_seconds": round(wall_clock, 2),
        "output_dir": summary.get("output_dir") or run_dir,
        "output_csv": summary.get("output_csv"),
        "effective_config": {"steps": cfg.steps, **{k: summary.get(k, v) for k, v in {**timing, "step_length": cfg.step_length}.items()}},
    }
    for k, d in [("total_steps", 0), ("observation_window_begin", None), ("observation_window_end", None),
                 ("observation_window_duration", None), ("safety_events", []), ("phase_change_count", 0),
                 ("phase_durations_per_tls", {}), ("reward_config", None), ("reward_config_source", None),
                 ("controlled_tls_ids", [])]:
        record[k] = summary.get(k, d)

    target_summary_dir = summary.get("output_dir") or run_dir
    if cfg.save_run_details and target_summary_dir:
        _write_json(os.path.join(target_summary_dir, "run_summary.json"), record)

    wait_label = f"{trip_wait:.2f}s" if trip_wait is not None else "N/A"
    print(f"==> {cfg.scenario_name} | {cfg.algorithm} | Seed {cfg.seed} | Wait: {wait_label} | Queue: {record['mean_queue_length']:.2f} | Time: {wall_clock:.1f}s")
    return (record, summary) if include_summary else record


def aggregate_benchmark_statistics(
    run_records: List[Dict[str, Any]],
    baseline_algo: str = "Fixed-Time",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Computes mean, std, and improvement percentages grouped by scenario and algorithm."""
    baseline_algo = normalize_algo_name(baseline_algo)
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    failed_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    cohorts, unique = {}, {}

    for record in run_records:
        norm_algo = normalize_algo_name(record.get("algorithm", ""))
        key = (record.get("scenario_name"), norm_algo)
        cohort = record.get("cohort_fingerprint")
        if key in cohorts and cohorts[key] != cohort:
            raise ValueError("Cannot aggregate different experiment cohorts")
        cohorts[key] = cohort
        unique[(*key, record.get("seed"))] = {**record, "algorithm": norm_algo}

    for r in unique.values():
        scen, algo = r.get("scenario_name", "unknown"), r.get("algorithm", "unknown")
        target = failed_groups if not r.get("is_valid", False) or r.get("status") == "FAILED" else groups
        target.setdefault(scen, {}).setdefault(algo, []).append(r)

    summary_stats: Dict[str, Dict[str, Dict[str, Any]]] = {scen: {} for scen in (groups.keys() | failed_groups.keys())}

    for scen, algos in groups.items():
        for algo, runs in algos.items():
            distinct_seeds = sorted({r["seed"] for r in runs if "seed" in r})
            algo_stats: Dict[str, Any] = {
                "effective_config": runs[0].get("effective_config"),
                "reward_config": runs[0].get("reward_config"),
                "controlled_tls_ids": runs[0].get("controlled_tls_ids", []),
                "run_count": len(runs),
                "successful_count": len(runs),
                "failed_count": 0,
                "seeds": distinct_seeds,
                "status": "SUCCESS",
                "is_valid": True,
            }
            for key, _ in (*BENCHMARK_METRICS, ("episode_reward", True)):
                vals = [float(r[key]) for r in runs if r.get(key) is not None and math.isfinite(float(r[key]))]
                algo_stats[key] = {
                    "mean": round(float(np.mean(vals)), 2) if vals else None,
                    "std": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else (0.0 if vals else None),
                    "min": round(min(vals), 2) if vals else None,
                    "max": round(max(vals), 2) if vals else None,
                    "available": bool(vals),
                }
                if key in ("collision_count", "teleport_count"):
                    algo_stats[key]["sum"] = int(sum(vals))

            col_sum = algo_stats.get("collision_count", {}).get("sum", 0)
            tel_sum = algo_stats.get("teleport_count", {}).get("sum", 0)
            algo_stats["safety_status"] = ("UNKNOWN" if not algo_stats["collision_count"]["available"] or not algo_stats["teleport_count"]["available"]
                                           else ("INCIDENTS_DETECTED" if col_sum or tel_sum else "SAFE"))
            cr_mean = algo_stats.get("completion_rate_pct", {}).get("mean", 100.0)
            algo_stats["service_status"] = ("NO_DEMAND" if all(r.get("total_demand") == 0 for r in runs)
                                            else service_status(cr_mean / 100.0 if cr_mean is not None else None))
            summary_stats[scen][algo] = algo_stats

        # Compute improvements relative to baseline algorithm
        base_stats = summary_stats[scen].get(baseline_algo)
        if base_stats and base_stats.get("is_valid", False):
            for algo, stats in summary_stats[scen].items():
                if algo == baseline_algo or not stats.get("is_valid", False):
                    continue
                b_tls = set(base_stats.get("controlled_tls_ids") or [])
                a_tls = set(stats.get("controlled_tls_ids") or [])
                if b_tls and a_tls and b_tls != a_tls:
                    stats["comparison_unavailable_reason"] = (
                        f"Controlled TLS mismatch in '{scen}': baseline '{baseline_algo}' controls {sorted(b_tls)}, "
                        f"but '{algo}' controls {sorted(a_tls)}. Cannot compare runs with differing control scopes."
                    )
                    continue
                b_runs = {r.get("seed"): r for r in algos[baseline_algo]}
                c_runs = {r.get("seed"): r for r in algos[algo]}
                if b_runs.keys() != c_runs.keys() or None in b_runs:
                    if failed_groups.get(scen, {}).get(algo) or failed_groups.get(scen, {}).get(baseline_algo):
                        stats["comparison_unavailable_reason"] = "Failed runs leave unequal successful seed sets"
                        continue
                    raise ValueError(f"Cannot compare '{algo}' with '{baseline_algo}' in '{scen}': seed sets differ or are missing")

                verified = True
                window_keys = ("total_steps", "observation_window_begin", "observation_window_end", "observation_window_duration")
                config_keys = ("steps", "step_length", *DEFAULT_SAFETY_CONFIG, "max_red")
                for seed, b_run in b_runs.items():
                    c_run = c_runs[seed]
                    b_cfg, c_cfg = b_run.get("effective_config") or {}, c_run.get("effective_config") or {}
                    b_cond = {k: b_cfg.get(k) for k in config_keys} | {k: b_run.get(k) for k in (*window_keys, "metric_schema_version")}
                    c_cond = {k: c_cfg.get(k) for k in config_keys} | {k: c_run.get(k) for k in (*window_keys, "metric_schema_version")}
                    if b_cond != c_cond:
                        raise ValueError(f"Cannot compare '{algo}' with '{baseline_algo}' in '{scen}', seed {seed}: simulation conditions differ")
                    if any(b_run.get(k) is None for k in window_keys) or "steps" not in b_cfg or b_cfg.get("step_length") is None:
                        verified = False
                if not verified:
                    stats["comparison_unavailable_reason"] = "Missing simulation window or step configuration"
                    continue

                improvements: Dict[str, float] = {}
                for key, higher_is_better in BENCHMARK_METRICS:
                    if key in base_stats and key in stats:
                        bm, am = base_stats[key].get("mean"), stats[key].get("mean")
                        if bm is not None and am is not None:
                            imp = compute_improvement(bm, am, higher_is_better=higher_is_better)
                            if imp is not None:
                                improvements[f"{key}_improvement_pct"] = round(imp, 2)
                stats["improvements_vs_baseline"] = improvements

    for scen, algorithms in failed_groups.items():
        for algo, failures in algorithms.items():
            stats = summary_stats[scen].setdefault(algo, {"run_count": 0, "seeds": [], "is_valid": False, "successful_count": 0})
            stats.update(
                failed_count=len(failures),
                failed_seeds=sorted(r["seed"] for r in failures),
                status="PARTIAL" if stats["run_count"] else "FAILED",
                error_messages=list(dict.fromkeys(r.get("error_message", "Simulation failed") for r in failures)),
            )

    return summary_stats


def run_full_benchmark_matrix(
    scenarios: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    config_file: str = "configs/config.yaml",
    output_dir: str = "results/evaluation",
    resume: bool = True,
    save_run_details: bool = False,
    generate_report: bool = True,
    generate_plots: bool = False,
    figures_dir: str = "results/figures",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs full benchmark matrix across scenarios, algorithms, and seeds."""
    yaml_cfg = load_experiments_yaml(config_file)
    sim_yaml = yaml_cfg.get("simulation", {})
    scenarios = scenarios or yaml_cfg.get("scenarios", ["cologne1", "ingolstadt1"])
    algorithms = [normalize_algo_name(a) for a in (algorithms or yaml_cfg.get("algorithms", ["Fixed-Time", "Max-Pressure"]))]
    seeds = seeds or yaml_cfg.get("seeds", [sim_yaml.get("seed", 42)])
    options = simulation_options(sim_yaml)

    os.makedirs(output_dir, exist_ok=True)
    master_file = os.path.join(output_dir, "benchmark_runs.json")
    all_records: List[Dict[str, Any]] = []

    if resume and os.path.isfile(master_file):
        try:
            all_records = json.loads(Path(master_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot resume invalid benchmark history: {master_file}; use --fresh") from error
        if not isinstance(all_records, list):
            raise ValueError(f"Benchmark history must be a list: {master_file}; use --fresh")

    executed = {r.get("run_fingerprint") for r in all_records if r.get("status") == "SUCCESS" and r.get("is_valid")}
    requested = set()

    for scen, algo, s in product(scenarios, algorithms, seeds):
        cfg = ExperimentConfig(scenario_name=scen, algorithm=algo, seed=s, output_dir=output_dir, **options, save_run_details=save_run_details)
        try:
            fp = compute_run_fingerprint(cfg)
            requested.add(fp)
            if resume and fp in executed:
                continue
            rec = run_single_experiment(cfg)
            rec.update(run_fingerprint=fp, cohort_fingerprint=compute_cohort_fingerprint(cfg))
        except Exception as err:
            fp = getattr(cfg, "run_id", f"{scen}_{algo}_{s}_failed")
            requested.add(fp)
            rec = {"scenario_name": scen, "algorithm": algo, "seed": s, "status": "FAILED", "is_valid": False, "error_message": str(err), "run_fingerprint": fp}
        all_records.append(rec)
        _write_json(master_file, all_records)

    selected = [r for r in all_records if r.get("run_fingerprint") in requested]
    summary_stats = aggregate_benchmark_statistics(selected, baseline_algo="Fixed-Time")
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    _write_json(summary_file, summary_stats)

    fields = list(dict.fromkeys(key for record in selected for key in record))
    with open(os.path.join(output_dir, "benchmark_runs.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    if generate_report:
        from src.evaluation.compare_algorithms import generate_comparison_report
        generate_comparison_report(eval_dir=output_dir, summary_file=summary_file, output_report=os.path.join(output_dir, "benchmark_comparison.md"))

    if generate_plots:
        from src.evaluation.plot_comparison import generate_all_benchmark_charts
        generate_all_benchmark_charts(output_dir=figures_dir, summary_file=summary_file, runs_file=master_file, selected_records=selected)

    return selected, summary_stats


def _parse_cli_items(raw_value: Optional[str], default_val: Any, cast=str) -> Optional[List[Any]]:
    if raw_value is None:
        return default_val
    parts = [p.strip() for p in raw_value.split(",") if p.strip()]
    return [cast(p) for p in parts] if parts else default_val


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Traffic Signal Benchmark Experiment Runner")
    parser.add_argument("--scenario", "--scenarios", dest="scenarios", type=str, default=None)
    parser.add_argument("--algorithm", "--algorithms", dest="algorithms", type=str, default=None)
    parser.add_argument("--seed", "--seeds", dest="seeds", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action="store_true", default=False)
    args = parser.parse_args(argv)

    scenarios = _parse_cli_items(args.scenarios, None if args.run_all else ["cologne1"])
    algorithms = _parse_cli_items(args.algorithms, None if args.run_all else ["Fixed-Time"], cast=normalize_algo_name)
    seeds = _parse_cli_items(args.seeds, None if args.run_all else [None], cast=int)

    is_multi = (
        args.run_all
        or (scenarios is not None and len(scenarios) > 1)
        or (algorithms is not None and len(algorithms) > 1)
        or (seeds is not None and len(seeds) > 1)
        or (args.report is False)
        or args.plots
        or args.fresh
    )

    if not is_multi:
        yaml_cfg = load_experiments_yaml(args.config)
        sim_yaml = yaml_cfg.get("simulation", {})
        seed_val = seeds[0] if (seeds and seeds[0] is not None) else sim_yaml.get("seed", 42)
        cfg = ExperimentConfig(
            scenario_name=scenarios[0] if scenarios else "cologne1",
            algorithm=algorithms[0] if algorithms else "Fixed-Time",
            seed=seed_val,
            **simulation_options(sim_yaml),
            save_run_details=args.save_details,
        )
        record = run_single_experiment(cfg)
        if record.get("status") == "FAILED" or not record.get("is_valid", True):
            sys.stderr.write(f"Experiment failed: {record.get('error_message', 'Unknown error')}\n")
            return 1
        return 0

    matrix_res = run_full_benchmark_matrix(
        scenarios=scenarios, algorithms=algorithms, seeds=seeds,
        config_file=args.config, resume=not args.fresh, save_run_details=args.save_details,
        generate_report=args.report, generate_plots=args.plots,
    )
    records = matrix_res[0] if isinstance(matrix_res, tuple) else matrix_res
    return 1 if any(r.get("status") == "FAILED" or not r.get("is_valid", True) for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
