"""Unit tests for src/data_inspector.py."""

import os
import tempfile
import pytest


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

    # Controlled lanes (incoming)
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


def test_extended_map_attributes_cologne1():
    """Tests extended map attributes: outgoing lanes, physical lanes, movements, and coordinates."""
    from src.data_inspector import inspect_scenario

    info = inspect_scenario("cologne1", data_dir="data")
    tls_id = info.tls_ids[0]

    # 1. Incoming and Outgoing lanes (Essential for Max-Pressure)
    in_lanes = info.get_incoming_lanes(tls_id)
    out_lanes = info.get_outgoing_lanes(tls_id)
    assert len(in_lanes) > 0
    assert len(out_lanes) > 0
    assert in_lanes == info.controlled_lanes_per_tls[tls_id]

    # 2. Movements / Connections mapping
    movements = info.get_movements(tls_id)
    assert len(movements) == 20
    # Verified link indices are 0 to 19 in order
    assert [m.link_index for m in movements] == list(range(20))
    # Direction check (should have straight 's', right 'r', left 'l')
    dirs = {m.direction for m in movements}
    assert "s" in dirs
    assert "r" in dirs
    assert "l" in dirs

    # 3. Physical lane properties (length and speed limit for Q-learning density/delay)
    assert len(info.lanes) > 0
    sample_lane_id = in_lanes[0]
    assert sample_lane_id in info.lanes
    assert info.get_lane_length(sample_lane_id) > 0.0
    assert info.get_lane_speed(sample_lane_id) > 0.0

    # 4. Spatial attributes & Boundary
    assert tls_id in info.junction_positions
    x, y = info.junction_positions[tls_id]
    assert isinstance(x, float) and isinstance(y, float)
    assert info.boundary is not None
    assert len(info.boundary) == 4
    assert info.boundary[0] < info.boundary[2]  # xmin < xmax
    assert info.boundary[1] < info.boundary[3]  # ymin < ymax


def test_multi_tls_cologne3():
    """Tests multi-intersection scenario metadata on cologne3."""
    from src.data_inspector import inspect_scenario

    info = inspect_scenario("cologne3", data_dir="data")
    assert info.name == "cologne3"
    assert len(info.tls_ids) == 3

    for tls_id in info.tls_ids:
        assert len(info.get_incoming_lanes(tls_id)) > 0
        assert len(info.get_outgoing_lanes(tls_id)) > 0
        assert len(info.get_movements(tls_id)) > 0
        assert len(info.get_green_phases(tls_id)) > 0
        assert tls_id in info.junction_positions


def test_flow_parsing_in_route():
    """Tests _parse_departures with simulated <flow> tags."""
    from src.data_inspector import _parse_departures

    xml_content = """<routes>
        <vehicle id="v1" depart="10.0"/>
        <trip id="t1" depart="25.0"/>
        <flow id="f1" begin="100.0" end="200.0" number="5"/>
        <flow id="f2" begin="300.0" end="360.0" period="20.0"/>
    </routes>"""

    with tempfile.NamedTemporaryFile("w", suffix=".rou.xml", delete=False) as f:
        f.write(xml_content)
        temp_path = f.name

    try:
        departures = _parse_departures(temp_path)
        assert len(departures) == 2 + 5 + 3  # 2 direct + 5 from f1 + 3 from f2 (300, 320, 340)
        assert departures[0] == 10.0
        assert departures[1] == 25.0
        assert 100.0 in departures
        assert 300.0 in departures
        assert 340.0 in departures
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_inspect_all_scenarios():
    """Tests inspect_all_scenarios finds and parses all benchmark scenarios."""
    from src.data_inspector import inspect_all_scenarios

    scenarios = inspect_all_scenarios("data")
    assert "cologne1" in scenarios
    assert "cologne3" in scenarios
    assert "ingolstadt1" in scenarios
