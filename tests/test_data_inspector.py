"""Unit tests for src/data_inspector.py."""

import pytest
import os


def test_inspect_cologne1():
    from src.data_inspector import inspect_scenario

    info = inspect_scenario("cologne1", data_dir="data")
    assert info.name == "cologne1"
    assert os.path.exists(info.cfg_path)
    assert os.path.exists(info.net_path)
    assert len(info.tls_ids) == 1
    assert info.tls_ids[0] == "cluster_357187_359543"

    tls_id = info.tls_ids[0]
    phases = info.phases_per_tls[tls_id]
    assert len(phases) == 8
    # Check that all states are 20 characters long
    for p in phases:
        assert len(p.state) == 20

    # Controlled lanes
    assert len(info.controlled_lanes_per_tls[tls_id]) > 0
    # Demand
    assert info.trips_or_vehicles_count > 0


def test_inspect_ingolstadt1():
    from src.data_inspector import inspect_scenario

    info = inspect_scenario("ingolstadt1", data_dir="data")
    assert info.name == "ingolstadt1"
    assert len(info.tls_ids) == 1
    assert info.tls_ids[0] == "cluster_274083968_cluster_1200364014_1200364088"

    tls_id = info.tls_ids[0]
    phases = info.phases_per_tls[tls_id]
    assert len(phases) == 6
    for p in phases:
        assert len(p.state) == 8

    # In ingolstadt1, phases 0, 2, 4 are green, phases 1, 3, 5 are yellow
    green_phases = [p.index for p in info.get_green_phases()]
    yellow_phases = [p.index for p in info.get_yellow_phases()]
    assert green_phases == [0, 2, 4]
    assert yellow_phases == [1, 3, 5]


def test_inspect_nonexistent_scenario():
    from src.data_inspector import inspect_scenario

    with pytest.raises(FileNotFoundError):
        inspect_scenario("non_existent_map", data_dir="data")


def test_get_window_demand():
    from src.data_inspector import get_window_demand

    demand_info = get_window_demand("cologne1", data_dir="data")
    assert "scheduled_demand_in_window" in demand_info
    assert demand_info["scheduled_demand_in_window"] > 0
    assert demand_info["window_duration"] == 3600.0
