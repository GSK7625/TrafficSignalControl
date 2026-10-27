"""Command-Line Interface for SUMO Traffic Signal Control Evaluation.

Usage examples:
  python main.py --scenario cologne1 --controller q_learning
  python main.py --all --controller max_pressure
  python main.py --compare
  python main.py --compare --scenarios cologne1 ingolstadt1
"""

import argparse
import os
import sys
from typing import List, Optional

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.evaluate import (
    DEFAULT_CONTROLLERS,
    DEFAULT_SCENARIOS,
    evaluate_comparison,
    evaluate_multiple,
    evaluate_scenario,
    export_evaluation_json,
    generate_algorithm_comparison_markdown,
    generate_comparison_markdown,
    plot_comparison_chart,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command line arguments.

    Args:
        argv: Command line argument list (defaults to sys.argv[1:] if None).

    Returns:
        Parsed arguments Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Đánh giá và so sánh thuật toán điều khiển đèn tín hiệu trên SUMO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Chỉ định tên kịch bản SUMO cần đánh giá (ví dụ: cologne1, ingolstadt1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Đánh giá toàn diện trên cả 2 bản đồ (cologne1 và ingolstadt1) và xuất báo cáo",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="fixed_time",
        choices=["fixed_time", "max_pressure", "q_learning"],
        help="Thuật toán điều khiển đèn tín hiệu: fixed_time, max_pressure, hoặc q_learning",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Chạy so sánh đối đầu đa thuật toán (Fixed-Time vs Max-Pressure) và xuất báo cáo đối sánh",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=DEFAULT_CONTROLLERS,
        help="Danh sách thuật toán cần so sánh khi dùng cờ --compare (mặc định: fixed_time max_pressure)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Khởi chạy giao diện trực quan sumo-gui thay vì chế độ dòng lệnh",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3600,
        help="Số bước mô phỏng tối đa (giây)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Thư mục lưu trữ kết quả JSON, XML và biểu đồ báo cáo",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Thư mục gốc chứa các kịch bản SUMO",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point for evaluation execution.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    args = parse_args(argv)

    if not args.all and not args.scenario and not args.compare:
        print("\n[!] Lỗi cú pháp: Vui lòng chỉ định kịch bản bằng:")
        print("    python main.py --scenario <tên_bản_đồ> [--controller q_learning]")
        print("    python main.py --all [--controller max_pressure]")
        print("    python main.py --compare")
        print("\nSử dụng `python main.py --help` để xem trợ giúp chi tiết.\n")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare:
        print("=================================================================")
        print(" BẮT ĐẦU BENCHMARK SO SÁNH ĐA THUẬT TOÁN ĐIỀU KHIỂN ĐÈN TÍN HIỆU")
        print(f" Danh sách thuật toán: {', '.join(args.controllers)}")
        target_scenarios = [args.scenario] if args.scenario else DEFAULT_SCENARIOS
        print(f" Danh sách bản đồ:     {', '.join(target_scenarios)}")
        print(f" Max steps: {args.max_steps}s | Chế độ GUI: {args.gui}")
        print("=================================================================\n")

        comp_results = evaluate_comparison(
            scenarios=target_scenarios,
            controllers=args.controllers,
            max_steps=args.max_steps,
            gui=args.gui,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
        )

        table_md = generate_algorithm_comparison_markdown(comp_results)
        print("\n" + table_md + "\n")

        json_file = os.path.join(args.output_dir, "comparison_evaluation.json")
        chart_file = os.path.join(args.output_dir, "algorithm_comparison.png")
        report_file = os.path.join(args.output_dir, "COMPARISON_REPORT.md")

        print("-----------------------------------------------------------------")
        print(f"[✓] Đã xuất tệp kết quả JSON:  {json_file}")
        print(f"[✓] Đã xuất biểu đồ trực quan: {chart_file}")
        print(f"[✓] Đã xuất báo cáo Markdown:  {report_file}")
        print("=================================================================\n")
        return 0

    if args.all:
        print("=================================================================")
        print(f" BẮT ĐẦU ĐÁNH GIÁ ĐA BẢN ĐỒ VỚI THUẬT TOÁN: {args.controller.upper()}")
        print(f" Danh sách bản đồ: {', '.join(DEFAULT_SCENARIOS)}")
        print(f" Max steps: {args.max_steps}s | Chế độ GUI: {args.gui}")
        print("=================================================================\n")

        results = evaluate_multiple(
            scenarios=DEFAULT_SCENARIOS,
            max_steps=args.max_steps,
            gui=args.gui,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            controller_type=args.controller,
        )

        # Print Markdown table to stdout
        table_md = generate_comparison_markdown(results)
        print("\n" + table_md + "\n")

        json_file = os.path.join(args.output_dir, f"{args.controller}_evaluation.json")
        chart_file = os.path.join(args.output_dir, f"{args.controller}_metrics.png")
        print("-----------------------------------------------------------------")
        print(f"[✓] Đã xuất tệp kết quả JSON:  {json_file}")
        print(f"[✓] Đã xuất biểu đồ trực quan: {chart_file}")
        print("=================================================================\n")
        return 0

    if args.scenario:
        print("=================================================================")
        print(f" BẮT ĐẦU ĐÁNH GIÁ KỊCH BẢN: {args.scenario} (Thuật toán: {args.controller})")
        print(f" Max steps: {args.max_steps}s | Chế độ GUI: {args.gui}")
        print("=================================================================\n")

        try:
            res = evaluate_scenario(
                scenario_name=args.scenario,
                max_steps=args.max_steps,
                gui=args.gui,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                controller_type=args.controller,
            )
        except Exception as err:
            print(f"[X] Lỗi trong quá trình mô phỏng {args.scenario}: {err}", file=sys.stderr)
            return 2

        results = {args.scenario: res}
        table_md = generate_comparison_markdown(results)
        print("\n" + table_md + "\n")

        json_file = os.path.join(args.output_dir, f"{args.controller}_evaluation.json")
        export_evaluation_json(results, output_path=json_file)

        chart_file = os.path.join(args.output_dir, f"{args.scenario}_{args.controller}_metrics.png")
        plot_comparison_chart(results, output_path=chart_file)

        print("-----------------------------------------------------------------")
        print(f"[✓] Đã xuất tệp kết quả JSON:  {json_file}")
        print(f"[✓] Đã xuất biểu đồ trực quan: {chart_file}")
        print("=================================================================\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
