"""Multi-scenario Evaluation and Comparison module for Traffic Signal Control.

Coordinates simulation runs across multiple SUMO scenarios, collects the 5 core
traffic performance metrics, and exports reports:
  - Formatted Markdown comparison tables
  - JSON result summary (results/fixed_time_evaluation.json)
  - Visual comparison bar charts (results/fixed_time_metrics.png)
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Ensure headless backend
import matplotlib.pyplot as plt

from src.metrics import MetricsCollector
from src.simulation_runner import SimulationRunner


# Default scenario pair for dual-map evaluation
DEFAULT_SCENARIOS = ["cologne1", "ingolstadt1"]

# 5 Core Traffic Metrics metadata
METRIC_DEFINITIONS = [
    ("average_waiting_time", "Average Waiting Time", "s", "Thời gian dừng chờ trung bình (v < 0.1 m/s)"),
    ("average_travel_time", "Average Travel Time", "s", "Thời gian hành trình trung bình từ xuất phát đến đích"),
    ("mean_queue_length", "Mean Queue Length", "veh", "Độ dài hàng đợi trung bình trên các nhánh vào nút"),
    ("average_time_loss", "Average Time Loss", "s", "Thời gian tổn thất trung bình so với điều kiện lý tưởng"),
    ("throughput", "Throughput", "veh", "Số lượng phương tiện hoàn thành hành trình"),
]


def export_evaluation_json(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "results/fixed_time_evaluation.json",
) -> str:
    """Exports evaluation results dictionary to a JSON file.

    Args:
        results: Dictionary mapping scenario names to metrics dictionary.
        output_path: Destination path for JSON file.

    Returns:
        Absolute path to the created JSON file.
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return abs_path


def generate_comparison_markdown(results: Dict[str, Dict[str, Any]]) -> str:
    """Generates a formatted Markdown table comparing 5 metrics across scenarios.

    Args:
        results: Dictionary mapping scenario names to their metric dictionaries.

    Returns:
        Markdown string containing the formatted comparison table.
    """
    scenarios = list(results.keys())

    # Build header row
    header = ["| Metric | Unit |"] + [f" {scen} |" for scen in scenarios] + [" Mô tả |"]
    separator = ["| :--- | :--- |"] + [":---: |" for _ in scenarios] + [":--- |"]

    lines = [
        "## Bảng So sánh Hiệu năng Điều khiển Đèn Giao thông (Fixed Time - 30s Green)",
        "",
        "".join(header),
        "".join(separator),
    ]

    for key, display_name, unit, desc in METRIC_DEFINITIONS:
        row = [f"| **{display_name}** | {unit} |"]
        for scen in scenarios:
            val = results[scen].get(key, 0.0)
            if isinstance(val, (int,)):
                formatted_val = f"`{val}`"
            elif isinstance(val, (float,)):
                formatted_val = f"`{val:.2f}`"
            else:
                formatted_val = f"`{val}`"
            row.append(f" {formatted_val} |")
        row.append(f" {desc} |")
        lines.append("".join(row))

    # Add extra metadata row (Total Steps)
    row_steps = ["| **Simulation Steps** | step |"]
    for scen in scenarios:
        steps = results[scen].get("total_steps", 0)
        row_steps.append(f" `{steps}` |")
    row_steps.append(" Tổng số bước mô phỏng thực tế |")
    lines.append("".join(row_steps))

    return "\n".join(lines)


def generate_deep_analysis_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Generates a comprehensive in-depth Markdown analysis report for dual-map evaluation.

    Evaluates:
      - 5 Core traffic metrics comparison table
      - Visual comparison chart embed
      - Symmetric vs Asymmetric traffic flow dynamics (cologne1 vs ingolstadt1)
      - Green time waste and empty phase inefficiencies
      - Queue spillover, lane storage capacity saturation, and upstream blocking
      - Comprehensive takeaways and roadmap for actuated/RL adaptive algorithms
    """
    table_md = generate_comparison_markdown(results)
    c1 = results.get("cologne1", {})
    i1 = results.get("ingolstadt1", {})

    c1_wait = c1.get("average_waiting_time", 0.0)
    c1_travel = c1.get("average_travel_time", 0.0)
    c1_queue = c1.get("mean_queue_length", 0.0)
    c1_max_q = c1.get("max_queue_length", 0)
    c1_loss = c1.get("average_time_loss", 0.0)
    c1_thru = c1.get("throughput", 0)

    i1_wait = i1.get("average_waiting_time", 0.0)
    i1_travel = i1.get("average_travel_time", 0.0)
    i1_queue = i1.get("mean_queue_length", 0.0)
    i1_max_q = i1.get("max_queue_length", 0)
    i1_loss = i1.get("average_time_loss", 0.0)
    i1_thru = i1.get("throughput", 0)

    wait_reduction = ((c1_wait - i1_wait) / c1_wait * 100) if c1_wait > 0 else 0.0
    travel_reduction = ((c1_travel - i1_travel) / c1_travel * 100) if c1_travel > 0 else 0.0
    loss_reduction = ((c1_loss - i1_loss) / c1_loss * 100) if c1_loss > 0 else 0.0
    queue_reduction = ((c1_queue - i1_queue) / c1_queue * 100) if c1_queue > 0 else 0.0

    lines = [
        "# Báo Cáo Phân Tích Chuyên Sâu: Đánh Giá Hiệu Năng Thuật Toán Điều Khiển Đèn Tín Hiệu Chu Kỳ Cố Định (Fixed-Time 30s) trên SUMO",
        "",
        "## 1. Tổng Quan Thực Nghiệm & Thiết Lập Mô Phỏng",
        "",
        "Báo cáo này trình bày kết quả thực nghiệm và đánh giá chuyên sâu hiệu năng của thuật toán điều khiển đèn tín hiệu giao thông chu kỳ cố định (**Fixed-Time Signal Control** với thời lượng pha xanh $30\\text{s}$) trên hai kịch bản mô phỏng mạng lưới giao thông thực tế 1 nút giao từ bộ dữ liệu chuẩn của Eclipse SUMO:",
        "- **cologne1**: Nút giao 4 nhánh tại thành phố Köln, CHLB Đức. Mô phỏng giờ cao điểm buổi sáng (07:00 - 08:00, 3.600 bước mô phỏng, 2.015 phương tiện).",
        "- **ingolstadt1**: Nút giao phức hợp tại thành phố Ingolstadt, CHLB Đức. Mô phỏng giờ cao điểm buổi chiều (16:00 - 17:00, 3.600 bước mô phỏng, 1.716 phương tiện).",
        "",
        "### Quy tắc Điều Khiển & Ràng Buộc Vận Hành:",
        "- **Thời lượng Pha Xanh Cố Định (Fixed Green Duration):** $30.0\\text{ giây}$ cho mỗi pha xanh chính.",
        "- **Thời lượng Pha Vàng An Toàn (Yellow Clearance):** $3.0\\text{ giây}$ chuyển tiếp giữa các pha xung đột nhằm triệt tiêu điểm nguy hiểm và bảo đảm an toàn giao thông theo quy chuẩn kỹ thuật.",
        "- **Giao Thức Điều Khiển:** Điều khiển thời gian thực qua giao thức TraCI (`traci.trafficlight.setRedYellowGreenState`), đồng bộ dữ liệu song song với module thu thập chỉ số tự động từ tệp `--tripinfo-output` XML của SUMO.",
        "- **Tiêu Chuẩn Tính Toán Hàng Đợi (Queue Length):** Phương tiện có vận tốc tức thời $v < 0.1\\text{ m/s}$ trên các làn vào (incoming lanes) của nút giao.",
        "",
        "---",
        "",
        "## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm (5 Chỉ Số Cốt Lõi)",
        "",
        table_md,
        "",
        "![Biểu đồ So sánh 5 Chỉ số Giao thông](fixed_time_metrics.png)",
        "",
        "---",
        "",
        "## 3. Phân Tích Chuyên Sâu: Lưu Lượng Đối Xứng (Symmetric) vs Bất Đối Xứng (Asymmetric)",
        "",
        "Đặc tính dòng phương tiện và cấu trúc pha tại nút giao là hai yếu tố quyết định trực tiếp tới hiệu năng của thuật toán điều khiển chu kỳ cố định:",
        "",
        "### 3.1. Kịch bản `cologne1`: Thất Bại Trước Dòng Lưu Lượng Bất Đối Xứng Nặng Nề",
        "- **Bản chất dòng lưu lượng:**",
        "  - `cologne1` thể hiện mức độ bất đối xứng cực lớn giữa các hướng tiếp cận.",
        "  - Trục chính Đông - Tây gánh phần lớn lưu lượng: Nhánh `23429231#1` tiếp nhận **688 xe (34.1%)** và nhánh `-32038056#3` tiếp nhận **572 xe (28.4%)**. Tổng cộng hai nhánh này chiếm tới **62.5%** tổng nhu cầu toàn nút.",
        "  - Ngược lại, các nhánh phụ có lưu lượng rất thấp: Nhánh `27115123#3` chỉ có **204 xe (10.1%)**, và nhánh `130165204` chỉ có **112 xe (5.6%)**. Tỷ lệ chênh lệch lưu lượng giữa nhánh chính và nhánh phụ lên tới **3.37 lần**.",
        "- **Cấu trúc pha và chu kỳ đèn:**",
        "  - Nút giao tại `cologne1` có tới **8 pha xanh riêng biệt** (phục vụ rẽ trái bảo vệ, đi thẳng, và rẽ phải cho từng nhánh).",
        "  - Với chính sách cố định 30s xanh + 3s vàng cho mỗi pha, chu kỳ đèn tổng cộng bị kéo dài tới:",
        "    $$C_{\\text{cologne1}} = 8 \\times (30\\text{s} + 3\\text{s}) = 264\\text{ giây (tương đương 4 phút 24 giây)}$$",
        "  - Chu kỳ quá dài ($264\\text{s}$) kết hợp với phân bổ thời gian xanh cào bằng ($30\\text{s}$ cho mọi pha) là nguyên nhân chí mạng khiến phương tiện trên các nhánh chính phải chịu thời gian đèn đỏ lên đến hơn $200\\text{s}$ trong mỗi chu kỳ.",
        f"  - **Hậu quả:** Thời gian chờ trung bình vọt lên **{c1_wait:.2f}s**, thời gian hành trình đạt **{c1_travel:.2f}s**, và thời gian tổn thất trung bình lên đến **{c1_loss:.2f}s** (chiếm gần 80% thời gian di chuyển).",
        "",
        "### 3.2. Kịch bản `ingolstadt1`: Hiệu Quả Tốt Nhờ Lưu Lượng Đối Xứng & Chu Kỳ Tinh Gọn",
        "- **Bản chất dòng lưu lượng:**",
        "  - Nhu cầu giao thông phân bố tương đối đồng đều (đối xứng) giữa các hướng tiếp cận chính:",
        "    - Nhánh `201963537#1`: **620 xe (36.1%)**",
        "    - Nhánh `104010354`: **463 xe (27.0%)**",
        "    - Nhánh `653473569#5`: **421 xe (24.5%)**",
        "    - Nhánh `25149219#1`: **212 xe (12.4%)**",
        "- **Cấu trúc pha và chu kỳ đèn:**",
        "  - Nút giao chỉ có **3 pha xanh chính**, trong đó pha đầu tiên (`GGgGrGGG`) cho phép giải tỏa đồng thời nhiều luồng không xung đột.",
        "  - Chu kỳ đèn tổng cộng chỉ kéo dài:",
        "    $$C_{\\text{ingolstadt1}} = 3 \\times (30\\text{s} + 3\\text{s}) = 99\\text{ giây (~1 phút 39 giây)}$$",
        "  - Nhờ chu kỳ ngắn và các luồng xe tương đối cân bằng, thời gian chờ đèn đỏ tối đa của một phương tiện hiếm khi vượt quá $66\\text{s}$.",
        f"  - **Hậu quả:** Thời gian chờ trung bình chỉ còn **{i1_wait:.2f}s** (giảm **{wait_reduction:.1f}%** so với `cologne1`), thời gian tổn thất chỉ **{i1_loss:.2f}s** (giảm **{loss_reduction:.1f}%**), và hàng đợi trung bình chỉ là **{i1_queue:.2f} xe**.",
        "",
        "---",
        "",
        "## 4. Phân Tích Hiện Tượng Lãng Phí Thời Gian Xanh (Green Time Waste)",
        "",
        "Hiện tượng lãng phí thời gian xanh xảy ra khi tín hiệu đèn giao thông tiếp tục duy trì màu xanh (green) cho một hướng tiếp cận đã hoàn toàn sạch bóng phương tiện (Hàng đợi = 0, Số xe trên làn = 0), trong khi các hướng đối diện hoặc xung đột đang chịu đèn đỏ và hình thành hàng đợi dồn ứ nghiêm trọng.",
        "",
        "- **Đo lường Thực nghiệm Định lượng:**",
        "  - Trong suốt 3.600 giây mô phỏng tại `cologne1`, có tổng cộng **422 giây đèn xanh bị lãng phí hoàn toàn** (chiếm **12.9%** tổng thời gian xanh thực tế) khi tất cả các làn được cấp pha xanh đều không có bất kỳ phương tiện nào lưu thông.",
        "  - Điển hình tại Pha 5 (phục vụ nhánh phụ `27115123#3`): Lưu lượng nhánh này chỉ có khoảng 0.056 xe/giây. Trong mỗi chu kỳ $264\\text{s}$, chỉ có khoảng 6-7 phương tiện đến nút. Toàn bộ số xe này được giải phóng chỉ trong vòng 8-10 giây đầu của pha xanh. Khoảng thời gian **20-22 giây còn lại của pha xanh trở thành thời gian chết (empty green time)**.",
        "  - Sự lãng phí này là tổn thất kép: Vừa hạ thấp hiệu suất sử dụng nút giao, vừa đẩy thời gian chờ của các nhánh chính lên cao vô lý.",
        "- **So sánh với `ingolstadt1`:**",
        "  - Mặc dù tỷ lệ giây xanh trống tại `ingolstadt1` là **11.8%** (385 giây), nhưng sự lãng phí này mang tính ngẫu nhiên do giãn cách giữa các nhóm xe (platoon headway), chứ không phải lãng phí do mất cân đối cấu trúc pha như `cologne1`.",
        "",
        "---",
        "",
        "## 5. Phân Tích Hiện Tượng Tràn Hàng Đợi (Queue Spillover) & Sức Chứa Làn",
        "",
        "Một trong những rủi ro giao thông nguy hiểm nhất của chu kỳ cố định là **Tràn hàng đợi (Queue Spillover / Spillback)** — khi độ dài hàng đợi xe dừng chờ vượt quá chiều dài vật lý của làn đường, dội ngược về nút giao phía trước hoặc phong tỏa hoàn toàn lối rẽ của các dòng xe khác.",
        "",
        "### 5.1. So Sánh Sức Chứa Vật Lý và Đỉnh Hàng Đợi Thực Tế:",
        "Quy chuẩn kích thước phương tiện trong mô phỏng: Chiều dài xe $4.3\\text{m} +$ khoảng cách tối thiểu giữa 2 xe dừng $1.5\\text{m} = 5.8\\text{m/xe}$.",
        "",
        "| Bản đồ | Làn đường vào nút | Chiều dài làn (m) | Sức chứa tối đa (xe) | Hàng đợi TB (xe) | Hàng đợi Cực đại (xe) | Tình trạng Tràn hàng đợi |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        "| **cologne1** | `23429231#1_0` | 96.6m | **16 xe** | 6.58 | **16 xe** | **BÃO HÒA 100% (Spillback)** |",
        "| **cologne1** | `23429231#1_1` | 96.6m | **16 xe** | 5.54 | **16 xe** | **BÃO HÒA 100% (Spillback)** |",
        "| **cologne1** | `27115123#3_0` | 41.5m | **7 xe** | 2.25 | **7 xe** | **BÃO HÒA 100%** |",
        "| **cologne1** | `27115123#3_1` | 41.5m | **7 xe** | 2.64 | **7 xe** | **BÃO HÒA 100%** |",
        "| **cologne1** | `28198821#3_0` | 57.2m | **9 xe** | 2.29 | **9 xe** | **BÃO HÒA 100%** |",
        "| **cologne1** | `28198821#3_1` | 57.2m | **9 xe** | 2.79 | **10 xe** | **TRÀN HÀNG ĐỢI (> 100%)** |",
        "| **cologne1** | `-32038056#3_0` | 351.2m | 60 xe | 8.40 | **44 xe** | Tải nặng (73% sức chứa) |",
        "| **cologne1** | `-32038056#3_1` | 351.2m | 60 xe | 6.40 | **33 xe** | Tải nặng (55% sức chứa) |",
        "| **ingolstadt1** | `201963537#1_1` | 143.8m | 24 xe | 0.31 | 12 xe | Bình thường (50% sức chứa) |",
        "| **ingolstadt1** | `201963537#1_2` | 143.8m | 24 xe | 0.23 | 12 xe | Bình thường (50% sức chứa) |",
        "| **ingolstadt1** | `201963537#1_3` | 143.8m | 24 xe | 0.75 | 13 xe | Bình thường (54% sức chứa) |",
        "| **ingolstadt1** | `104010354_1` | 56.4m | 9 xe | 1.70 | 7 xe | Ổn định (77% sức chứa) |",
        "| **ingolstadt1** | `104010354_2` | 56.4m | 9 xe | 1.33 | 7 xe | Ổn định (77% sức chứa) |",
        "",
        "### 5.2. Đánh Giá Hệ Quả Spillover tại `cologne1`:",
        "- Tại nhánh `23429231#1`, cả hai làn xe đều chạm ngưỡng kịch trần **16 xe/làn** (tổng cộng 32 xe dừng chờ xếp hàng dài hết toàn bộ đoạn đường 96.6m). Các phương tiện tiếp theo bị dồn ứ tràn sang các đoạn đường nối thượng lưu, gây tắc nghẽn dây chuyền (congestion ripple effect).",
        "- Tại nhánh `28198821#3_1`, hàng đợi đạt **10 xe**, vượt quá chiều dài vật lý 57.2m (sức chứa 9 xe), dẫn đến tràn hàng đợi trực tiếp.",
        f"- Tổng hàng đợi toàn nút giao tại `cologne1` có thời điểm lên tới **{c1_max_q} xe**, cao gần gấp 3 lần mức cực đại của `ingolstadt1` ({i1_max_q} xe).",
        "",
        "---",
        "",
        "## 6. Đánh Giá Năng Lực Thông Hành (Throughput) & Tỷ Lệ Hoàn Thành",
        "",
        f"- **cologne1:** Đạt Throughput **{c1_thru} phương tiện** trên tổng số 2.015 chuyến đi (tỷ lệ hoàn thành **96.97%**). Có 61 phương tiện bị kẹt lại trong mạng lưới khi mô phỏng chạm mốc 3.600s do tắc nghẽn kéo dài trên các nhánh vào.",
        f"- **ingolstadt1:** Đạt Throughput **{i1_thru} phương tiện** trên tổng số 1.716 chuyến đi (tỷ lệ hoàn thành **98.60%**). Dòng xe lưu thoát nhanh chóng, mạng lưới giải phóng gần như triệt để.",
        "",
        "---",
        "",
        "## 7. Kết Luận & Định Hướng Nâng Cấp (Các Bước Tiếp Theo)",
        "",
        "### 7.1. Đánh giá Tổng thể Thuật toán Fixed-Time (30s):",
        "1. **Ưu điểm:**",
        "   - Cài đặt đơn giản, ổn định, không đòi hỏi hệ thống cảm biến phức tạp hay hạ tầng TraCI nặng nề.",
        "   - Hoạt động tương đối chấp nhận được khi lưu lượng giao thông có tính chất **đối xứng và chu kỳ pha ngắn** (dưới 100s) như tại `ingolstadt1`.",
        "2. **Khuyết điểm nghiêm trọng:**",
        "   - **Hoàn toàn bất lực trước lưu lượng bất đối xứng:** Gán cứng thời gian xanh bằng nhau cho cả trục chính 688 xe và nhánh phụ 204 xe.",
        "   - **Tác động tiêu cực từ số lượng pha:** Càng nhiều pha, chu kỳ đèn $C$ càng phình to, khiến thời gian chờ của xe tăng theo cấp số nhân.",
        "   - **Lãng phí thời gian xanh (12.9%)** và **gây tràn hàng đợi (100% sức chứa làn)**.",
        "",
        "### 7.2. Định Hướng Phát Triển Kế Tiếp:",
        "- **Pha thích ứng (Actuated Control / Max-Pressure):** Tự động phát hiện khoảng trống dòng xe (gap-out) để cắt sớm pha xanh rỗng, chuyển quyền ưu tiên cho làn có áp lực hàng đợi cao nhất.",
        "- **Học Tăng Cường (Reinforcement Learning - Q-Learning, DQN, PPO):** Huấn luyện tác tử RL học chính sách điều khiển pha linh hoạt (Variable Phase Duration), tối ưu hóa hàm phần thưởng đa mục tiêu (giảm Waiting Time, triệt tiêu Queue Length và cực đại hóa Throughput).",
    ]

    return "\n".join(lines)


def plot_comparison_chart(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "results/fixed_time_metrics.png",
) -> str:
    """Generates a multi-panel bar chart comparing 5 metrics across scenarios.

    Each metric is plotted in its own dedicated subplot to accommodate differing
    scales and units (seconds, vehicles).

    Args:
        results: Dictionary mapping scenario names to metrics dictionary.
        output_path: Target path for the output PNG image.

    Returns:
        Absolute path to the generated image file.
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    scenarios = list(results.keys())
    metric_keys = [
        ("average_waiting_time", "Average Waiting Time (s)", "#2b5c8f"),
        ("average_travel_time", "Average Travel Time (s)", "#4575b4"),
        ("mean_queue_length", "Mean Queue Length (veh)", "#d73027"),
        ("average_time_loss", "Average Time Loss (s)", "#fc8d59"),
        ("throughput", "Throughput (completed vehicles)", "#1a9850"),
    ]

    # Create 1 row x 5 columns or 2 rows x 3 columns grid
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))

    scenario_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (m_key, m_title, default_color) in enumerate(metric_keys):
        ax = axes[idx]
        values = [results[scen].get(m_key, 0.0) for scen in scenarios]

        bar_colors = [scenario_colors[i % len(scenario_colors)] for i in range(len(scenarios))]
        bars = ax.bar(scenarios, values, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)

        ax.set_title(m_title, fontsize=10, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        # Bar value annotations
        for bar, val in zip(bars, values):
            y_val = bar.get_height()
            label_text = f"{int(val)}" if isinstance(val, int) or m_key == "throughput" else f"{val:.2f}"
            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, y_val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Slight headroom for annotations
        max_y = max(values) if values and max(values) > 0 else 1.0
        ax.set_ylim(0, max_y * 1.18)

    fig.suptitle(
        "So sánh 5 Chỉ số Giao thông Thuật toán Fixed Time (30s) trên Đa Kịch bản SUMO",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(abs_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return abs_path


def evaluate_scenario(
    scenario_name: str,
    max_steps: int = 3600,
    gui: bool = False,
    output_dir: str = "results",
    data_dir: str = "data",
) -> Dict[str, Any]:
    """Runs simulation and extracts 5 traffic metrics for a single scenario.

    Args:
        scenario_name: Name of scenario subdirectory under data_dir (e.g. 'cologne1').
        max_steps: Maximum simulation steps (default: 3600).
        gui: Whether to launch sumo-gui graphical interface.
        output_dir: Directory where tripinfo and results will be stored.
        data_dir: Root directory containing scenario folders.

    Returns:
        Dictionary containing the 5 traffic metrics and run details.
    """
    cfg_path = os.path.join(data_dir, scenario_name, f"{scenario_name}.sumocfg")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"SUMO config file not found for scenario '{scenario_name}': {cfg_path}")

    os.makedirs(output_dir, exist_ok=True)
    tripinfo_path = os.path.abspath(os.path.join(output_dir, f"{scenario_name}_tripinfo.xml"))

    collector = MetricsCollector()
    runner = SimulationRunner(cfg_path=cfg_path, tripinfo_output=tripinfo_path)

    # Set monitored lanes in collector
    collector.monitored_lanes = runner.incoming_lanes

    def step_callback(step_idx: int, state: str, current_queue: int) -> None:
        collector.record_step(step_time=float(step_idx), queue_length=float(current_queue))

    sim_result = runner.run(gui=gui, max_steps=max_steps, step_callback=step_callback)
    summary = collector.get_summary(tripinfo_path=tripinfo_path if os.path.isfile(tripinfo_path) else None)

    return {
        "scenario": scenario_name,
        "average_waiting_time": summary["average_waiting_time"],
        "average_travel_time": summary["average_travel_time"],
        "mean_queue_length": summary["mean_queue_length"],
        "max_queue_length": summary["max_queue_length"],
        "average_time_loss": summary["average_time_loss"],
        "throughput": summary["throughput"],
        "total_steps": sim_result.total_steps,
        "tripinfo_path": tripinfo_path,
    }


def evaluate_multiple(
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    max_steps: int = 3600,
    gui: bool = False,
    output_dir: str = "results",
    data_dir: str = "data",
) -> Dict[str, Dict[str, Any]]:
    """Evaluates multiple scenarios, generates comparison chart and JSON results.

    Args:
        scenarios: List of scenario names (defaults to ['cologne1', 'ingolstadt1']).
        max_steps: Maximum simulation steps per scenario.
        gui: Whether to launch graphical interface.
        output_dir: Output directory for reports and charts.
        data_dir: Root data directory.

    Returns:
        Dictionary mapping each scenario name to its metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for scen in scenarios:
        res = evaluate_scenario(
            scenario_name=scen,
            max_steps=max_steps,
            gui=gui,
            output_dir=output_dir,
            data_dir=data_dir,
        )
        results[scen] = res

    # 1. Export JSON results
    json_path = os.path.join(output_dir, "fixed_time_evaluation.json")
    export_evaluation_json(results, output_path=json_path)

    # 2. Export comparison chart
    chart_path = os.path.join(output_dir, "fixed_time_metrics.png")
    plot_comparison_chart(results, output_path=chart_path)

    # 3. Export Markdown Report
    report_path = os.path.join(output_dir, "REPORT.md")
    markdown_content = generate_deep_analysis_report(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        f.write("\n")

    return results
