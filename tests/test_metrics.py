"""Unit tests for MetricsCollector and tripinfo output parser."""

import os
from unittest.mock import MagicMock
import pytest

from src.metrics import MetricsCollector, StepMetrics, parse_tripinfo_output


class TestParseTripinfoOutput:
    """Test suite for parse_tripinfo_output function."""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_tripinfo_output("non_existent_file.xml")

    def test_empty_tripinfo_xml(self, tmp_path):
        xml_path = tmp_path / "empty.xml"
        xml_path.write_text("<tripinfos></tripinfos>", encoding="utf-8")
        res = parse_tripinfo_output(str(xml_path))
        assert res["tripinfo_completed_vehicles"] == 0
        assert res["tripinfo_mean_waiting_time"] is None
        assert res["tripinfo_total_waiting_time"] == 0.0

    def test_valid_tripinfo_xml(self, tmp_path):
        xml_path = tmp_path / "valid.xml"
        xml_content = """<tripinfos>
    <tripinfo id="veh_1" duration="50.0" waitingTime="10.0" timeLoss="15.0"/>
    <tripinfo id="veh_2" duration="70.0" waitingTime="20.0" timeLoss="25.0"/>
</tripinfos>"""
        xml_path.write_text(xml_content, encoding="utf-8")
        res = parse_tripinfo_output(str(xml_path))
        assert res["tripinfo_completed_vehicles"] == 2
        assert res["tripinfo_mean_travel_time"] == 60.0
        assert res["tripinfo_mean_waiting_time"] == 15.0
        assert res["tripinfo_total_waiting_time"] == 30.0
        assert res["tripinfo_mean_time_loss"] == 20.0
        assert res["tripinfo_total_time_loss"] == 40.0


class TestMetricsCollector:
    """Test suite for MetricsCollector class."""

    def test_empty_summary(self):
        collector = MetricsCollector(scenario_name="test_scen")
        summary = collector.get_summary()
        assert summary["total_steps"] == 0
        assert summary["total_completed_vehicles"] == 0
        assert summary["is_valid"] is True
        assert summary["status"] == "OK"
        assert "core_metrics" in summary
        assert summary["core_metrics"]["queue_length"]["mean_queue_length"] == 0.0
        assert summary["core_metrics"]["waiting_time"]["mean_waiting_time"] is None
        assert summary["core_metrics"]["throughput"]["throughput_vph"] == 0.0

    def test_record_step_with_mock_traci(self, mock_traci):
        collector = MetricsCollector(scenario_name="test_scen")
        mock_traci.simulation.getDepartedIDList.return_value = ["v1", "v2"]
        mock_traci.vehicle.getDeparture.side_effect = lambda vid: 0.0 if vid == "v1" else 1.0
        mock_traci.lane.getLastStepHaltingNumber.return_value = 2
        mock_traci.vehicle.getIDList.return_value = ["v1", "v2"]
        mock_traci.vehicle.getAccumulatedWaitingTime.return_value = 5.0
        mock_traci.vehicle.getTimeLoss.return_value = 6.0

        step_record = collector.record_step(
            step=1,
            sim_time=1.0,
            controlled_lanes=["lane_1"],
            traci_module=mock_traci,
            step_length=1.0,
        )

        assert isinstance(step_record, StepMetrics)
        assert step_record.step == 1
        assert step_record.running_vehicles == 2
        assert step_record.queue_length == 2
        assert step_record.waiting_time == 10.0  # 5.0 * 2 vehs
        assert step_record.time_loss == 12.0     # 6.0 * 2 vehs

    def test_export_results_safety_and_structure(self, tmp_path):
        collector = MetricsCollector(scenario_name="test_scen")
        collector.step_records.append(
            StepMetrics(
                step=1,
                sim_time=1.0,
                running_vehicles=1,
                queue_length=1,
                waiting_time=2.0,
                time_loss=3.0,
                step_completed_vehicles=0,
                cumulative_completed_vehicles=0,
            )
        )

        json_path, csv_path = collector.export_results(str(tmp_path))
        assert os.path.isfile(json_path)
        assert os.path.isfile(csv_path)

        data_abs = os.path.abspath("data")
        with pytest.raises(ValueError, match="Cannot save results into data directory"):
            collector.export_results(os.path.join(data_abs, "forbidden"))

    def test_get_core_metrics(self):
        collector = MetricsCollector(scenario_name="test_scen")
        collector.step_records.append(
            StepMetrics(
                step=1,
                sim_time=1.0,
                running_vehicles=2,
                queue_length=3,
                waiting_time=4.5,
                step_completed_vehicles=1,
                cumulative_completed_vehicles=1,
            )
        )
        core = collector.get_core_metrics()
        assert "mean_waiting_time" in core
        assert core["mean_queue_length"] == 3.0
        assert core["max_queue_length"] == 3
        assert core["total_completed_vehicles"] == 1
        assert "throughput_vph" in core

        summary = collector.get_summary()
        assert "core_metrics" in summary
        assert summary["core_metrics"]["queue_length"]["mean_queue_length"] == 3.0

    def test_required_metric_failure_is_not_a_zero_queue(self, mock_traci):
        mock_traci.lane.getLastStepHaltingNumber.side_effect = RuntimeError("connection lost")
        collector = MetricsCollector("audit")
        with pytest.raises(RuntimeError, match="connection lost"):
            collector.record_step(1, 1.0, ["lane"], mock_traci)
        assert collector.step_records == []

    @pytest.mark.parametrize("xml,status", [
        ("<tripinfos>broken", "PARSE_ERROR"),
        (None, "MISSING"),
    ])
    def test_tripinfo_reconciliation_does_not_claim_false_match(self, tmp_path, xml, status):
        path = tmp_path / "tripinfo.xml"
        if xml is not None:
            path.write_text(xml, encoding="utf-8")
        collector = MetricsCollector("audit", str(path))
        collector._cumulative_completed = 1
        assert collector.reconcile_tripinfo()["tripinfo_reconciliation_status"] == status

    def test_early_summary_does_not_cache_missing_tripinfo(self, tmp_path):
        path = tmp_path / "tripinfo.xml"
        collector = MetricsCollector("audit", str(path))
        collector.get_summary()
        path.write_text("<tripinfos/>", encoding="utf-8")
        assert collector.reconcile_tripinfo()["tripinfo_reconciliation_status"] == "MATCH"

    def test_nonfinite_tripinfo_is_invalid(self, tmp_path):
        path = tmp_path / "tripinfo.xml"
        path.write_text('<tripinfos><tripinfo id="a" duration="nan" waitingTime="nan" timeLoss="nan"/></tripinfos>', encoding="utf-8")
        collector = MetricsCollector("audit", str(path))
        collector._cumulative_completed = 1
        assert collector.reconcile_tripinfo()["tripinfo_reconciliation_status"] == "PARSE_ERROR"

    def test_no_arrivals_has_unavailable_waiting_mean(self, tmp_path):
        path = tmp_path / "tripinfo.xml"
        path.write_text("<tripinfos/>", encoding="utf-8")
        collector = MetricsCollector("audit", str(path))
        collector.reconcile_tripinfo()
        assert collector.get_summary()["tripinfo_mean_waiting_time"] is None
