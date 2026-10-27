"""Unit tests for MetricsCollector (Traffic metrics collection & tripinfo parsing)."""

import os
import tempfile
from unittest.mock import MagicMock
import pytest

from src.metrics import MetricsCollector, StepRecord


SAMPLE_TRIPINFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <tripinfo id="veh_0" depart="0.00" arrival="60.00" duration="60.00" waitingTime="15.00" timeLoss="20.00" routeLength="500.00"/>
    <tripinfo id="veh_1" depart="10.00" arrival="80.00" duration="70.00" waitingTime="25.00" timeLoss="30.00" routeLength="600.00"/>
    <tripinfo id="veh_2" depart="20.00" arrival="100.00" duration="80.00" waitingTime="20.00" timeLoss="25.00" routeLength="700.00"/>
</tripinfos>
"""

EMPTY_TRIPINFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
</tripinfos>
"""

MIXED_TRIPINFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <!-- Non-tripinfo element should be ignored -->
    <personinfo id="person_0" depart="5.00"/>
    <tripinfo id="veh_single" depart="0.00" arrival="50.00" duration="50.00" waitingTime="10.00" timeLoss="12.50" routeLength="400.00"/>
</tripinfos>
"""


@pytest.fixture
def sample_tripinfo_file():
    """Creates a temporary sample tripinfo XML file."""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_TRIPINFO_XML)
        f_path = f.name
    yield f_path
    if os.path.exists(f_path):
        os.remove(f_path)


@pytest.fixture
def empty_tripinfo_file():
    """Creates a temporary empty tripinfo XML file."""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(EMPTY_TRIPINFO_XML)
        f_path = f.name
    yield f_path
    if os.path.exists(f_path):
        os.remove(f_path)


@pytest.fixture
def mixed_tripinfo_file():
    """Creates a temporary mixed tripinfo XML file with personinfo and comments."""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(MIXED_TRIPINFO_XML)
        f_path = f.name
    yield f_path
    if os.path.exists(f_path):
        os.remove(f_path)


# ---------------------------------------------------------------------------
# Test Step Recording and Queue Length Calculations
# ---------------------------------------------------------------------------

def test_collector_initial_state():
    """MetricsCollector starts with empty queue history and 0.0 queue metrics."""
    collector = MetricsCollector()
    assert collector.total_steps == 0
    assert collector.queue_history == []
    assert collector.mean_queue_length == 0.0
    assert collector.max_queue_length == 0


def test_collector_record_step():
    """Test recording individual simulation steps."""
    collector = MetricsCollector()

    collector.record_step(step_time=0.0, queue_length=4, halted_vehicles=4, running_vehicles=10)
    collector.record_step(step_time=1.0, queue_length=8, halted_vehicles=8, running_vehicles=12)

    assert collector.total_steps == 2
    assert collector.queue_history == [4, 8]
    assert len(collector.step_records) == 2
    assert collector.step_records[0] == StepRecord(step_time=0.0, queue_length=4.0, halted_vehicles=4, running_vehicles=10)
    assert collector.step_records[1] == StepRecord(step_time=1.0, queue_length=8.0, halted_vehicles=8, running_vehicles=12)


def test_collector_mean_and_max_queue_length():
    """Test calculation of mean and max queue length over varying steps."""
    collector = MetricsCollector()
    steps = [0, 2, 4, 6]
    for idx, q in enumerate(steps):
        collector.record_step(step_time=float(idx), queue_length=q)

    # (0 + 2 + 4 + 6) / 4 = 3.0
    assert collector.mean_queue_length == 3.0
    assert collector.max_queue_length == 6
    assert collector.total_steps == 4


# ---------------------------------------------------------------------------
# Test Parsing Tripinfo XML
# ---------------------------------------------------------------------------

def test_parse_tripinfo_sample_data(sample_tripinfo_file):
    """Test parsing a valid tripinfo XML file containing 3 vehicle trips."""
    collector = MetricsCollector()
    metrics = collector.parse_tripinfo(sample_tripinfo_file)

    assert metrics["throughput"] == 3
    # Duration: (60 + 70 + 80) / 3 = 70.0
    assert pytest.approx(metrics["average_travel_time"], 0.001) == 70.0
    # WaitingTime: (15 + 25 + 20) / 3 = 20.0
    assert pytest.approx(metrics["average_waiting_time"], 0.001) == 20.0
    # TimeLoss: (20 + 30 + 25) / 3 = 25.0
    assert pytest.approx(metrics["average_time_loss"], 0.001) == 25.0
    # Totals
    assert pytest.approx(metrics["total_travel_time"], 0.001) == 210.0
    assert pytest.approx(metrics["total_waiting_time"], 0.001) == 60.0
    assert pytest.approx(metrics["total_time_loss"], 0.001) == 75.0


def test_parse_tripinfo_empty_data(empty_tripinfo_file):
    """Empty tripinfo file (0 vehicles completed) should return zero without division by zero."""
    collector = MetricsCollector()
    metrics = collector.parse_tripinfo(empty_tripinfo_file)

    assert metrics["throughput"] == 0
    assert metrics["average_travel_time"] == 0.0
    assert metrics["average_waiting_time"] == 0.0
    assert metrics["average_time_loss"] == 0.0
    assert metrics["total_travel_time"] == 0.0
    assert metrics["total_waiting_time"] == 0.0
    assert metrics["total_time_loss"] == 0.0


def test_parse_tripinfo_mixed_elements(mixed_tripinfo_file):
    """Non-tripinfo elements (e.g. personinfo) and comments must be ignored."""
    collector = MetricsCollector()
    metrics = collector.parse_tripinfo(mixed_tripinfo_file)

    assert metrics["throughput"] == 1
    assert pytest.approx(metrics["average_travel_time"], 0.001) == 50.0
    assert pytest.approx(metrics["average_waiting_time"], 0.001) == 10.0
    assert pytest.approx(metrics["average_time_loss"], 0.001) == 12.50


def test_parse_tripinfo_file_not_found():
    """Attempting to parse a nonexistent file should raise FileNotFoundError."""
    collector = MetricsCollector()
    with pytest.raises(FileNotFoundError):
        collector.parse_tripinfo("nonexistent_path/tripinfo.xml")


def test_parse_tripinfo_invalid_xml():
    """Corrupted or invalid XML content should raise ValueError."""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write("<tripinfos><unclosed_tag>")
        corrupted_path = f.name

    try:
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="Failed to parse XML"):
            collector.parse_tripinfo(corrupted_path)
    finally:
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)


# ---------------------------------------------------------------------------
# Test Comprehensive 5 Metrics Summary
# ---------------------------------------------------------------------------

def test_get_summary_combines_queue_and_tripinfo(sample_tripinfo_file):
    """get_summary() returns the 5 primary metrics combining TraCI queue & XML tripinfo."""
    collector = MetricsCollector()
    for q in [2, 4, 6]:
        collector.record_step(step_time=0.0, queue_length=q)

    summary = collector.get_summary(tripinfo_path=sample_tripinfo_file)

    # 1. Average Waiting Time
    assert "average_waiting_time" in summary
    assert pytest.approx(summary["average_waiting_time"], 0.001) == 20.0

    # 2. Average Travel Time
    assert "average_travel_time" in summary
    assert pytest.approx(summary["average_travel_time"], 0.001) == 70.0

    # 3. Queue Length (mean and max)
    assert "mean_queue_length" in summary
    assert pytest.approx(summary["mean_queue_length"], 0.001) == 4.0
    assert summary["max_queue_length"] == 6

    # 4. Average Time Loss
    assert "average_time_loss" in summary
    assert pytest.approx(summary["average_time_loss"], 0.001) == 25.0

    # 5. Throughput
    assert "throughput" in summary
    assert summary["throughput"] == 3

    assert summary["total_steps"] == 3
    assert summary["queue_history"] == [2, 4, 6]


def test_get_summary_without_tripinfo():
    """get_summary() when tripinfo hasn't been parsed returns default 0.0 for XML metrics."""
    collector = MetricsCollector()
    collector.record_step(step_time=0.0, queue_length=10)

    summary = collector.get_summary()
    assert summary["mean_queue_length"] == 10.0
    assert summary["max_queue_length"] == 10
    assert summary["throughput"] == 0
    assert summary["average_waiting_time"] == 0.0
    assert summary["average_travel_time"] == 0.0
    assert summary["average_time_loss"] == 0.0


def test_collector_reset(sample_tripinfo_file):
    """reset() should restore collector back to empty initial state."""
    collector = MetricsCollector()
    collector.record_step(step_time=1.0, queue_length=5)
    collector.parse_tripinfo(sample_tripinfo_file)

    assert collector.total_steps == 1
    assert collector.tripinfo_metrics["throughput"] == 3

    collector.reset()

    assert collector.total_steps == 0
    assert collector.queue_history == []
    assert collector.mean_queue_length == 0.0
    assert collector.max_queue_length == 0
    assert collector.tripinfo_metrics == {}


def test_metrics_integration_with_real_sumo_simulation():
    """Integration test connecting MetricsCollector with SimulationRunner and real SUMO."""
    from src.simulation_runner import SimulationRunner

    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    try:
        collector = MetricsCollector()
        runner = SimulationRunner(cfg_path=cfg_path, tripinfo_output=tripinfo_path)

        def on_step(step_idx: int, state: str, queue: int):
            collector.record_step(step_time=float(step_idx), queue_length=queue)

        runner.run(gui=False, max_steps=60, step_callback=on_step)

        summary = collector.get_summary(tripinfo_path=tripinfo_path)

        assert summary["total_steps"] == 60
        assert len(summary["queue_history"]) == 60
        assert summary["mean_queue_length"] >= 0.0
        assert summary["max_queue_length"] >= 0
        assert "throughput" in summary
        assert "average_waiting_time" in summary
        assert "average_travel_time" in summary
        assert "average_time_loss" in summary
    finally:
        if os.path.exists(tripinfo_path):
            os.remove(tripinfo_path)

