"""Multi-Seed Benchmark Comparison for Traffic Signal Control Algorithms.

Compares 3 controllers:
  1. Fixed-Time (Baseline)
  2. Max-Pressure
  3. Q-Learning (Trained policy in pure greedy evaluation mode)

Evaluates performance across multiple random seeds on SUMO scenarios
and aggregates statistics (mean ± std) for the 5 core traffic metrics:
  1. Average Waiting Time (s)
  2. Average Travel Time (s)
  3. Mean Queue Length (veh)
  4. Average Time Loss (s)
  5. Throughput (veh)
"""

import argparse
import json
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.controllers import FixedTimeController, MaxPressureController, QLearningController
from src.data_inspector import inspect_scenario
from src.metrics import MetricsCollector
from src.simulation_runner import SimulationRunner


def calculate_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Calculates sample mean and standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = float(sum(values) / n)
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = float(math.sqrt(max(0.0, variance)))
    return mean, std


def run_single_simulation(
    cfg_path: str,
    controller_type: str,
    seed: int,
    max_steps: int = 3600,
    model_path: Optional[str] = None,
    min_green: float = 20.0,
    max_green: float = 45.0,
    use_libsumo: bool = True,
    temp_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Runs a single simulation run with a given controller and seed.

    Args:
        cfg_path: Path to SUMO .sumocfg file.
        controller_type: 'fixed_time', 'max_pressure', or 'q_learning'.
        seed: Random seed passed to SUMO via --seed.
        max_steps: Maximum simulation steps.
        model_path: Path to trained Q-table JSON (required if controller_type is 'q_learning').
        min_green: Minimum green phase duration in seconds (default: 20.0s).
        max_green: Maximum green phase duration in seconds (default: 45.0s).
        use_libsumo: True to use libsumo if available.
        temp_dir: Directory to store temporary tripinfo XML files.

    Returns:
        Dictionary of metrics for this single run.
    """
    clean_temp_dir = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="sumo_bench_")
        clean_temp_dir = True

    tripinfo_file = os.path.join(temp_dir, f"tripinfo_{controller_type}_seed_{seed}.xml")
    collector = MetricsCollector()

    try:
        scenario_name = os.path.splitext(os.path.basename(cfg_path))[0]
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(cfg_path)))
        info = inspect_scenario(scenario_name, data_dir=data_dir)
        tls_id = info.tls_ids[0]
        green_phases = info.get_green_phases(tls_id)
        movements = info.get_movements(tls_id)

        controller = None
        if controller_type == "fixed_time":
            controller = FixedTimeController(
                green_phases=green_phases,
                green_time=30.0,
                yellow_time=3.0,
                all_red_time=1.0,
                max_red=90.0,
                tls_id=tls_id,
            )
        elif controller_type == "max_pressure":
            controller = MaxPressureController(
                green_phases=green_phases,
                movements=movements,
                min_green=min_green,
                max_green=max_green,
                yellow_time=3.0,
                all_red_time=1.0,
                max_red=90.0,
                tls_id=tls_id,
            )
        elif controller_type == "q_learning":
            controller = QLearningController(
                green_phases=green_phases,
                movements=movements,
                min_green=min_green,
                max_green=max_green,
                yellow_time=3.0,
                all_red_time=1.0,
                max_red=90.0,
                decision_interval=5.0,
                tls_id=tls_id,
            )
            if model_path and os.path.isfile(model_path):
                controller.load(model_path)
                # Ensure evaluation respects configured min_green / max_green
                controller.min_green = min_green
                controller.max_green = max_green
            # Crucial: Disable exploration for evaluation
            controller.explore_mode = False
            controller.epsilon = 0.0

        runner = SimulationRunner(
            cfg_path=cfg_path,
            controller=controller,
            tls_id=tls_id,
            tripinfo_output=tripinfo_file,
            additional_args=["--seed", str(seed)],
            use_libsumo=use_libsumo,
        )

        def step_callback(step_idx: int, state: str, queue_len: int) -> None:
            collector.record_step(step_time=float(step_idx), queue_length=float(queue_len))

        sim_res = runner.run(gui=False, max_steps=max_steps, step_callback=step_callback)
        summary = collector.get_summary(tripinfo_path=tripinfo_file)

        return {
            "average_waiting_time": float(summary.get("average_waiting_time", 0.0)),
            "average_travel_time": float(summary.get("average_travel_time", 0.0)),
            "mean_queue_length": float(sim_res.mean_queue_length),
            "max_queue_length": float(sim_res.max_queue_length),
            "average_time_loss": float(summary.get("average_time_loss", 0.0)),
            "throughput": float(summary.get("throughput", 0)),
        }
    finally:
        if os.path.isfile(tripinfo_file):
            try:
                os.remove(tripinfo_file)
            except Exception:
                pass
        if clean_temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


def run_benchmark_matrix(
    scenario_name: str,
    data_dir: str = "data",
    algorithms: Sequence[str] = ("fixed_time", "max_pressure", "q_learning"),
    seeds: Sequence[int] = (42, 101, 2024, 777, 999),
    max_steps: int = 3600,
    min_green: float = 20.0,
    max_green: float = 45.0,
    model_path: Optional[str] = None,
    use_libsumo: bool = True,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs full benchmark matrix comparing algorithms across seeds.

    Args:
        scenario_name: Scenario directory name (e.g. 'cologne1').
        data_dir: Base directory containing scenario folders.
        algorithms: List of algorithms to run.
        seeds: Sequence of random seeds to evaluate.
        max_steps: Max simulation steps per run.
        min_green: Minimum green phase duration in seconds (default: 20.0s).
        max_green: Maximum green phase duration in seconds (default: 45.0s).
        model_path: Path to trained Q-table JSON for Q-Learning.
        use_libsumo: Whether to use libsumo backend.
        output_dir: Directory where results will be saved.
        verbose: Print progress to stdout.

    Returns:
        Structured dictionary containing raw runs, aggregated stats, and comparisons.
    """
    info = inspect_scenario(scenario_name, data_dir=data_dir)
    cfg_path = info.cfg_path

    if "q_learning" in algorithms and (model_path is None or not os.path.isfile(model_path)):
        default_model = os.path.join("models", "q_learning", f"q_learning_{scenario_name}_best.json")
        fallback_model = os.path.join("models", "q_learning", f"q_learning_{scenario_name}.json")
        if os.path.isfile(default_model):
            model_path = default_model
        elif os.path.isfile(fallback_model):
            model_path = fallback_model

    raw_results: Dict[str, Dict[int, Dict[str, float]]] = {algo: {} for algo in algorithms}

    metric_keys = [
        "average_waiting_time",
        "average_travel_time",
        "mean_queue_length",
        "max_queue_length",
        "average_time_loss",
        "throughput",
    ]

    for seed in seeds:
        for algo in algorithms:
            if verbose:
                print(f"[Benchmark] Scenario: {scenario_name} | Seed: {seed:5d} | Controller: {algo:<12}", end=" ... ", flush=True)

            m_path = model_path if algo == "q_learning" else None
            metrics = run_single_simulation(
                cfg_path=cfg_path,
                controller_type=algo,
                seed=seed,
                max_steps=max_steps,
                model_path=m_path,
                min_green=min_green,
                max_green=max_green,
                use_libsumo=use_libsumo,
            )
            raw_results[algo][seed] = metrics
            if verbose:
                print(f"Wait: {metrics['average_waiting_time']:6.2f}s | Queue: {metrics['mean_queue_length']:5.2f} | TP: {int(metrics['throughput']):4d}")

    # Aggregate stats across seeds
    aggregated_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for algo in algorithms:
        aggregated_stats[algo] = {}
        for m_key in metric_keys:
            vals = [raw_results[algo][s][m_key] for s in seeds]
            mean_val, std_val = calculate_mean_std(vals)
            aggregated_stats[algo][m_key] = {
                "mean": round(mean_val, 3),
                "std": round(std_val, 3),
            }

    # Compute improvements relative to Fixed-Time baseline (if present)
    improvements: Dict[str, Dict[str, float]] = {}
    if "fixed_time" in aggregated_stats:
        baseline = aggregated_stats["fixed_time"]
        for algo in algorithms:
            if algo == "fixed_time":
                continue
            improvements[algo] = {}
            for m_key in metric_keys:
                b_mean = baseline[m_key]["mean"]
                a_mean = aggregated_stats[algo][m_key]["mean"]
                if m_key == "throughput":
                    # For throughput, higher is better
                    imp = ((a_mean - b_mean) / max(b_mean, 1e-6)) * 100.0
                else:
                    # For delay/waiting/queue, lower is better
                    imp = ((b_mean - a_mean) / max(b_mean, 1e-6)) * 100.0
                improvements[algo][m_key] = round(imp, 2)

    benchmark_data = {
        "scenario": scenario_name,
        "seeds": list(seeds),
        "algorithms": list(algorithms),
        "max_steps": max_steps,
        "min_green": min_green,
        "max_green": max_green,
        "model_path": model_path,
        "raw_results": {algo: {str(s): m for s, m in s_map.items()} for algo, s_map in raw_results.items()},
        "aggregated_stats": aggregated_stats,
        "improvements_vs_fixed_time": improvements,
    }

    # Format Markdown Report
    report_md = generate_benchmark_report_markdown(benchmark_data)

    # Save to output_dir
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"benchmark_{scenario_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    md_path = os.path.join(output_dir, f"benchmark_{scenario_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    if verbose:
        print(f"\n[Benchmark] Results saved to: {json_path}")
        print(f"[Benchmark] Markdown report saved to: {md_path}\n")

    return benchmark_data


def generate_benchmark_report_markdown(data: Dict[str, Any]) -> str:
    """Generates a clean Markdown table comparing algorithms across seeds."""
    scen = data["scenario"]
    seeds = data["seeds"]
    algos = data["algorithms"]
    stats = data["aggregated_stats"]
    improvements = data.get("improvements_vs_fixed_time", {})

    lines = [
        f"# BÁO CÁO SO SÁNH HIỆU NĂNG ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG",
        "",
        f"- **Kịch bản:** `{scen}`",
        f"- **Số lượng hạt giống (Seeds):** `{len(seeds)}` hạt giống ({seeds})",
        f"- **Thời gian xanh tối thiểu (min_green):** `{data.get('min_green', 20.0)}s`",
        f"- **Mô hình Q-Learning:** `{data.get('model_path') or 'N/A'}` (Chế độ Greedy Evaluation)",
        f"- **Thời gian mô phỏng:** `{data.get('max_steps')}s` mỗi lượt",
        "",
        "## Bảng tổng hợp các chỉ số hiệu năng (Mean ± Std)",
        "",
        "| Thuật toán | Thời gian chờ TB (s) | Thời gian di chuyển TB (s) | Độ dài hàng đợi TB (xe) | Tổn thất thời gian TB (s) | Thông lượng Throughput (xe) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    algo_labels = {
        "fixed_time": "Fixed-Time (Cố định)",
        "max_pressure": "Max-Pressure (Áp suất cực đại)",
        "q_learning": "Q-Learning (Đã hội tụ)",
    }

    for algo in algos:
        label = algo_labels.get(algo, algo)
        a_stats = stats.get(algo, {})
        wt = f"{a_stats.get('average_waiting_time', {}).get('mean', 0.0):.2f} ± {a_stats.get('average_waiting_time', {}).get('std', 0.0):.2f}"
        tt = f"{a_stats.get('average_travel_time', {}).get('mean', 0.0):.2f} ± {a_stats.get('average_travel_time', {}).get('std', 0.0):.2f}"
        ql = f"{a_stats.get('mean_queue_length', {}).get('mean', 0.0):.2f} ± {a_stats.get('mean_queue_length', {}).get('std', 0.0):.2f}"
        tl = f"{a_stats.get('average_time_loss', {}).get('mean', 0.0):.2f} ± {a_stats.get('average_time_loss', {}).get('std', 0.0):.2f}"
        tp = f"{a_stats.get('throughput', {}).get('mean', 0.0):.1f} ± {a_stats.get('throughput', {}).get('std', 0.0):.1f}"

        lines.append(f"| **{label}** | {wt} | {tt} | {ql} | {tl} | {tp} |")

    if improvements:
        lines.append("")
        lines.append("## Mức độ cải thiện so với Fixed-Time Baseline (%)")
        lines.append("")
        lines.append("*(Giá trị dương % thể hiện thuật toán hoạt động tốt hơn Fixed-Time)*")
        lines.append("")
        lines.append("| Thuật toán | Giảm TG Chờ (%) | Giảm TG Di chuyển (%) | Giảm Hàng đợi (%) | Giảm Tổn thất TG (%) | Tăng Thông lượng (%) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for algo in algos:
            if algo == "fixed_time":
                continue
            label = algo_labels.get(algo, algo)
            imp = improvements.get(algo, {})
            wt_imp = f"{imp.get('average_waiting_time', 0.0):+.2f}%"
            tt_imp = f"{imp.get('average_travel_time', 0.0):+.2f}%"
            ql_imp = f"{imp.get('mean_queue_length', 0.0):+.2f}%"
            tl_imp = f"{imp.get('average_time_loss', 0.0):+.2f}%"
            tp_imp = f"{imp.get('throughput', 0.0):+.2f}%"

            lines.append(f"| **{label}** | {wt_imp} | {tt_imp} | {ql_imp} | {tl_imp} | {tp_imp} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running multi-seed benchmarks."""
    parser = argparse.ArgumentParser(description="Multi-Seed Traffic Signal Benchmark Runner")
    parser.add_argument("--scenario", type=str, default="cologne1", help="Scenario name (e.g. cologne1, ingolstadt1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024, 777, 999], help="List of random seeds")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["fixed_time", "max_pressure", "q_learning"], help="Algorithms to benchmark")
    parser.add_argument("--max-steps", type=int, default=3600, help="Max simulation steps per run")
    parser.add_argument("--min-green", type=float, default=20.0, help="Minimum green phase duration in seconds (default: 20.0)")
    parser.add_argument("--max-green", type=float, default=45.0, help="Maximum green phase duration in seconds (default: 45.0)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained Q-Learning model JSON")
    parser.add_argument("--output-dir", type=str, default="results/benchmark", help="Directory for output files")
    parser.add_argument("--use-libsumo", dest="use_libsumo", action="store_true", default=True, help="Use libsumo backend")
    parser.add_argument("--no-libsumo", dest="use_libsumo", action="store_false", help="Use standard TraCI backend")

    args = parser.parse_args()

    run_benchmark_matrix(
        scenario_name=args.scenario,
        algorithms=args.algorithms,
        seeds=args.seeds,
        max_steps=args.max_steps,
        min_green=args.min_green,
        max_green=args.max_green,
        model_path=args.model,
        use_libsumo=args.use_libsumo,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
