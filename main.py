"""Command-Line Interface for SUMO Fixed-Time Traffic Signal Control Evaluation.

Usage examples:
  python main.py --scenario cologne1
  python main.py --scenario ingolstadt1 --gui
  python main.py --all
  python main.py --all --max-steps 3600
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
    DEFAULT_SCENARIOS,
    evaluate_multiple,
    evaluate_scenario,
    export_evaluation_json,
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
        description="Đánh giá thuật toán điều khiển đèn tín hiệu Fixed Time (30s) trên SUMO",
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
        help="Đánh giá toàn diện trên cả 2 bản đồ (cologne1 và ingolstadt1) và xuất báo cáo so sánh",
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

    if not args.all and not args.scenario:
        print("\n[!] Lỗi cú pháp: Vui lòng chỉ định kịch bản bằng:")
        print("    python main.py --scenario <tên_bản_đồ>  (ví dụ: python main.py --scenario cologne1)")
        print("  hoặc:")
        print("    python main.py --all                    (chạy cả 2 bản đồ cologne1 và ingolstadt1)")
        print("\nSử dụng `python main.py --help` để xem trợ giúp chi tiết.\n")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        print("=================================================================")
        print(" BẮT ĐẦU ĐÁNH GIÁ ĐA BẢN ĐỒ VỚI THUẬT TOÁN FIXED TIME (30S GREEN)")
        print(f" Danh sách bản đồ: {', '.join(DEFAULT_SCENARIOS)}")
        print(f" Max steps: {args.max_steps}s | Chế độ GUI: {args.gui}")
        print("=================================================================\n")

        results = evaluate_multiple(
            scenarios=DEFAULT_SCENARIOS,
            max_steps=args.max_steps,
            gui=args.gui,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
        )

        # Print Markdown table to stdout
        table_md = generate_comparison_markdown(results)
        print("\n" + table_md + "\n")

        json_file = os.path.join(args.output_dir, "fixed_time_evaluation.json")
        chart_file = os.path.join(args.output_dir, "fixed_time_metrics.png")
        report_file = os.path.join(args.output_dir, "REPORT.md")

        print("-----------------------------------------------------------------")
        print(f"[✓] Đã xuất tệp kết quả JSON:  {json_file}")
        print(f"[✓] Đã xuất biểu đồ trực quan: {chart_file}")
        print(f"[✓] Đã xuất báo cáo Markdown:  {report_file}")
        print("=================================================================\n")
        return 0

    if args.scenario:
        print("=================================================================")
        print(f" BẮT ĐẦU ĐÁNH GIÁ KỊCH BẢN: {args.scenario}")
        print(f" Max steps: {args.max_steps}s | Chế độ GUI: {args.gui}")
        print("=================================================================\n")

        try:
            res = evaluate_scenario(
                scenario_name=args.scenario,
                max_steps=args.max_steps,
                gui=args.gui,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
            )
        except Exception as err:
            print(f"[X] Lỗi trong quá trình mô phỏng {args.scenario}: {err}", file=sys.stderr)
            return 2

        # Wrap in dictionary for reporting
        results = {args.scenario: res}
        table_md = generate_comparison_markdown(results)
        print("\n" + table_md + "\n")

        # Save single or update json
        json_file = os.path.join(args.output_dir, "fixed_time_evaluation.json")
        export_evaluation_json(results, output_path=json_file)

        chart_file = os.path.join(args.output_dir, f"{args.scenario}_metrics.png")
        plot_comparison_chart(results, output_path=chart_file)

        print("-----------------------------------------------------------------")
        print(f"[✓] Đã xuất tệp kết quả JSON:  {json_file}")
        print(f"[✓] Đã xuất biểu đồ trực quan: {chart_file}")
        print("=================================================================\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
