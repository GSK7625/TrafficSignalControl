"""So sánh đối chuẩn đa hạt giống (Multi-Seed Benchmark) cho các thuật toán điều khiển tín hiệu đèn giao thông.

So sánh 3 bộ điều khiển:
  1. Fixed-Time (Baseline - Chu kỳ cố định)
  2. Max-Pressure (Áp suất cực đại)
  3. Q-Learning (Chính sách đã huấn luyện ở chế độ đánh giá thuần khai thác Greedy)

Đánh giá hiệu năng qua nhiều hạt giống ngẫu nhiên (random seeds) trên các kịch bản SUMO
và tổng hợp thống kê (mean ± std) cho 5 chỉ số giao thông cốt lõi:
  1. Thời gian chờ trung bình (Average Waiting Time - s)
  2. Thời gian di chuyển trung bình (Average Travel Time - s)
  3. Độ dài hàng đợi trung bình (Mean Queue Length - xe)
  4. Tổn thất thời gian trung bình (Average Time Loss - s)
  5. Thông lượng (Throughput - xe)
"""

import argparse
import json
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.controllers import build_default_controller
from src.data_inspector import inspect_scenario
from src.metrics import MetricsCollector
from src.simulation_runner import SimulationRunner


def calculate_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Tính giá trị trung bình mẫu và độ lệch chuẩn mẫu."""
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
    scenario_info: Optional[Any] = None,
) -> Dict[str, float]:
    """Chạy một lượt mô phỏng đơn lẻ với một bộ điều khiển và hạt giống ngẫu nhiên (seed) xác định.

    Tham số:
        cfg_path: Đường dẫn tới file .sumocfg của SUMO.
        controller_type: 'fixed_time', 'max_pressure', hoặc 'q_learning'.
        seed: Hạt giống ngẫu nhiên truyền vào SUMO qua cờ --seed.
        max_steps: Số bước mô phỏng tối đa.
        model_path: Đường dẫn tới file JSON Q-table đã huấn luyện (bắt buộc nếu là 'q_learning').
        min_green: Thời gian xanh tối thiểu tính bằng giây (mặc định: 20.0s).
        max_green: Thời gian xanh tối đa tính bằng giây (mặc định: 45.0s).
        use_libsumo: True nếu muốn dùng backend libsumo nếu có sẵn.
        temp_dir: Thư mục tạm lưu các file XML tripinfo.
        scenario_info: Đối tượng ScenarioInfo đọc sẵn để tránh đọc lại metadata mạng lưới.

    Trả về:
        Dictionary chứa các chỉ số đo được trong lượt chạy này.
    """
    clean_temp_dir = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="sumo_bench_")
        clean_temp_dir = True

    tripinfo_file = os.path.join(temp_dir, f"tripinfo_{controller_type}_seed_{seed}.xml")
    collector = MetricsCollector()

    try:
        info = scenario_info
        if info is None:
            scenario_name = os.path.splitext(os.path.basename(cfg_path))[0]
            data_dir = os.path.dirname(os.path.dirname(os.path.abspath(cfg_path)))
            info = inspect_scenario(scenario_name, data_dir=data_dir)
        tls_id = info.tls_ids[0]
        green_phases = info.get_green_phases(tls_id)
        movements = info.get_movements(tls_id)

        controller = build_default_controller(
            controller_type=controller_type,
            green_phases=green_phases,
            movements=movements,
            min_green=min_green,
            max_green=max_green,
            tls_id=tls_id,
        )
        if controller_type == "q_learning":
            if model_path and os.path.isfile(model_path):
                controller.load(model_path)
                controller.min_green = min_green
                controller.max_green = max_green
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
    """Chạy ma trận đối chuẩn so sánh các thuật toán qua nhiều hạt giống ngẫu nhiên (seeds).

    Tham số:
        scenario_name: Tên thư mục kịch bản (ví dụ: 'cologne1').
        data_dir: Thư mục gốc chứa các thư mục kịch bản.
        algorithms: Danh sách các thuật toán cần đánh giá.
        seeds: Tập hợp các hạt giống ngẫu nhiên (seeds) để đánh giá.
        max_steps: Số bước mô phỏng tối đa cho mỗi lượt chạy.
        min_green: Thời gian xanh tối thiểu tính bằng giây (mặc định: 20.0s).
        max_green: Thời gian xanh tối đa tính bằng giây (mặc định: 45.0s).
        model_path: Đường dẫn tới file JSON Q-table đã huấn luyện cho Q-Learning.
        use_libsumo: Có sử dụng backend libsumo hay không.
        output_dir: Thư mục lưu kết quả đối chuẩn.
        verbose: In tiến trình chi tiết ra stdout.

    Trả về:
        Dictionary cấu trúc chứa kết quả từng lượt chạy, số liệu tổng hợp và mức độ cải thiện.
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
                scenario_info=info,
            )
            raw_results[algo][seed] = metrics
            if verbose:
                print(f"Wait: {metrics['average_waiting_time']:6.2f}s | Queue: {metrics['mean_queue_length']:5.2f} | TP: {int(metrics['throughput']):4d}")

    # Tổng hợp số liệu thống kê qua các hạt giống (seeds)
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

    # Tính toán mức độ cải thiện so với baseline Fixed-Time (nếu có)
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
                    # Đối với thông lượng (throughput), giá trị càng cao càng tốt
                    imp = ((a_mean - b_mean) / max(b_mean, 1e-6)) * 100.0
                else:
                    # Đối với thời gian chờ/di chuyển/hàng đợi, giá trị càng thấp càng tốt
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

    # Định dạng báo cáo Markdown
    report_md = generate_benchmark_report_markdown(benchmark_data)

    # Lưu kết quả vào output_dir
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
    """Tạo bảng báo cáo Markdown sạch đẹp so sánh các thuật toán qua các hạt giống ngẫu nhiên."""
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
    """Điểm vào CLI để chạy đối chuẩn ma trận đa hạt giống."""
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
