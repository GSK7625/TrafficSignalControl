"""Kiểm thử bộ thu thập metrics và vòng mô phỏng (MetricsCollector & SimulationRunner).

Bao gồm:
  1. Đối chiếu tính tay ví dụ số nhỏ từ đặc tả Giai đoạn 2.
  2. Trường hợp biên quan trọng: không có xe hoàn thành (kiểm tra không ghi số 0 giả).
  3. Kiểm tra logic chuyển pha dọn nút giao an toàn (vàng và đỏ toàn phần).
  4. Chạy mô phỏng thực tế với TraCI trên cả cologne1 và ingolstadt1.
"""

import os
import tempfile
import pytest

from src.metrics import MetricsCollector
from src.simulation_runner import build_all_red_state, build_yellow_state, run_simulation


def test_hand_calculation_comparison():
    """Đối chiếu tuyệt đối với ví dụ tính tay trong đặc tả Giai đoạn 2."""
    incoming_lanes = ["lane_1", "lane_2"]
    collector = MetricsCollector(incoming_lanes=incoming_lanes, total_demand=4)

    # 1. Ghi nhận 5 bước với hàng đợi tính tay: [3, 4, 3, 1, 0]
    collector.record_step(sim_time=1.0, step_queue=3, running_count=2, inserted_step=2)
    collector.record_step(sim_time=2.0, step_queue=4, running_count=3, inserted_step=1)
    collector.record_step(sim_time=3.0, step_queue=3, running_count=4, inserted_step=1)
    collector.record_step(sim_time=4.0, step_queue=1, running_count=2, arrived_step=1)
    collector.record_step(sim_time=5.0, step_queue=0, running_count=1, arrived_step=2)

    # 2. Tạo file tripinfo.xml giả lập với đúng 3 xe hoàn thành:
    # v1: duration=3, waitingTime=1, timeLoss=1.5
    # v2: duration=4, waitingTime=2, timeLoss=2.8
    # v3: duration=3, waitingTime=0, timeLoss=0.5
    xml_content = """<tripinfos>
        <tripinfo id="v1" depart="1.0" arrival="4.0" duration="3.0" waitingTime="1.0" timeLoss="1.5"/>
        <tripinfo id="v2" depart="1.0" arrival="5.0" duration="4.0" waitingTime="2.0" timeLoss="2.8"/>
        <tripinfo id="v3" depart="2.0" arrival="5.0" duration="3.0" waitingTime="0.0" timeLoss="0.5"/>
    </tripinfos>"""

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        summary = collector.get_summary(
            tripinfo_path=temp_xml_path,
            begin_time=0.0,
            end_time=5.0,
        )

        # 3. Đối chiếu số liệu tính tay:
        # Mean Queue = (3 + 4 + 3 + 1 + 0) / 5 = 2.2 xe
        assert summary["mean_queue_length"] == 2.2
        # Max Queue = max(3, 4, 3, 1, 0) = 4 xe
        assert summary["max_queue_length"] == 4
        # Throughput = 3 xe
        assert summary["throughput"] == 3
        # Average Travel Time = (3 + 4 + 3) / 3 = 3.33 s
        assert summary["average_travel_time"] == 3.33
        # Average Waiting Time = (1 + 2 + 0) / 3 = 1.00 s
        assert summary["average_waiting_time"] == 1.00
        # Average Time Loss = (1.5 + 2.8 + 0.5) / 3 = 1.60 s
        assert summary["average_time_loss"] == 1.60

        # Kiểm tra thống kê xe
        assert summary["total_demand"] == 4
        assert summary["completed_vehicles"] == 3
        assert summary["running_vehicles_at_end"] == 1
        assert summary["completion_rate_pct"] == 75.0

    finally:
        if os.path.exists(temp_xml_path):
            os.remove(temp_xml_path)


def test_empty_completed_vehicles_edge_case():
    """Kiểm tra trường hợp biên: 0 xe hoàn thành (không được ghi số 0.0 giả)."""
    collector = MetricsCollector(incoming_lanes=["lane_1"], total_demand=10)
    collector.record_step(sim_time=1.0, step_queue=5, running_count=5, inserted_step=5)

    xml_content = """<tripinfos>
    </tripinfos>"""

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_xml_path = f.name

    try:
        summary = collector.get_summary(
            tripinfo_path=temp_xml_path,
            begin_time=0.0,
            end_time=10.0,
        )

        # Throughput = 0
        assert summary["throughput"] == 0
        # Average của tập rỗng PHẢI là None (không được ghi 0.0)
        assert summary["average_travel_time"] is None
        assert summary["average_waiting_time"] is None
        assert summary["average_time_loss"] is None

    finally:
        if os.path.exists(temp_xml_path):
            os.remove(temp_xml_path)


def test_yellow_and_all_red_builders():
    """Kiểm tra tính an toàn của hàm sinh đèn vàng và đỏ toàn phần."""
    # Khi chuyển giữa 2 pha đối kháng
    curr_state = "GGGrrr"
    next_state = "rrrGGG"
    yellow_state = build_yellow_state(curr_state, next_state)
    all_red_state = build_all_red_state(curr_state)

    assert yellow_state == "yyyrrr"
    assert all_red_state == "rrrrrr"

    # Khi có tín hiệu tiếp tục xanh ở pha sau
    p_curr = "GGrr"
    p_next = "GrrG"
    assert build_yellow_state(p_curr, p_next, only_changing=True) == "Gyrr"


def test_simulation_runner_short_run_cologne1():
    """Kiểm tra chạy mô phỏng ngắn (70 bước = 2 chu kỳ chuyển pha) trên cologne1."""
    out_dir = os.path.join("results", "tests", "cologne1_test")
    summary = run_simulation(
        scenario_name="cologne1",
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=1.0,
        seed=42,
        max_steps=70,  # 30 xanh + 3 vàng + 1 đỏ + 30 xanh + 3 vàng + 1 đỏ + 2 xanh = 70s
        gui=False,
        output_dir=out_dir,
    )

    assert summary["total_simulation_steps"] == 70
    assert summary["mean_queue_length"] is not None
    assert summary["max_queue_length"] is not None

    # Kiểm tra tồn tại đầy đủ các file đầu ra thô
    assert os.path.isfile(os.path.join(out_dir, "tripinfo.xml"))
    assert os.path.isfile(os.path.join(out_dir, "summary.json"))
    assert os.path.isfile(os.path.join(out_dir, "phase_log.json"))
    assert os.path.isfile(os.path.join(out_dir, "run_config.json"))


def test_simulation_runner_short_run_ingolstadt1():
    """Kiểm tra chạy mô phỏng ngắn (70 bước) trên ingolstadt1."""
    out_dir = os.path.join("results", "tests", "ingolstadt1_test")
    summary = run_simulation(
        scenario_name="ingolstadt1",
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=1.0,
        seed=42,
        max_steps=70,
        gui=False,
        output_dir=out_dir,
    )

    assert summary["total_simulation_steps"] == 70
    assert summary["mean_queue_length"] is not None
    assert os.path.isfile(os.path.join(out_dir, "summary.json"))


def test_simulation_runner_zero_completed_edge_case():
    """Kiểm tra chạy mô phỏng 2 bước (chưa có xe nào đến đích) trên cologne1."""
    out_dir = os.path.join("results", "tests", "zero_test")
    summary = run_simulation(
        scenario_name="cologne1",
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=1.0,
        seed=42,
        max_steps=2,  # Xe đầu tiên xuất phát ở giây thứ 5 (25205), nên ở 2 giây đầu không có xe
        gui=False,
        output_dir=out_dir,
    )

    assert summary["throughput"] == 0
    assert summary["average_travel_time"] is None
    assert summary["average_waiting_time"] is None
    assert summary["average_time_loss"] is None
