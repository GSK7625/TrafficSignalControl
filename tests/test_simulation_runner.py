"""Unit and integration tests for SimulationRunner (TraCI)."""

import os
import tempfile
import pytest
import traci

from src.controllers.fixed_time import FixedTimeController
from src.data_inspector import inspect_scenario
from src.simulation_runner import SimulationRunner, SimulationResult


def test_simulation_runner_initialization():
    """Test SimulationRunner initialization with default and custom parameters."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    runner = SimulationRunner(cfg_path=cfg_path)

    assert runner.cfg_path == os.path.abspath(cfg_path)
    assert runner.tls_id == "cluster_357187_359543"
    assert runner.controller is not None
    assert isinstance(runner.controller, FixedTimeController)
    assert runner.step_duration == 1.0
    assert len(runner.incoming_lanes) > 0


def test_simulation_runner_sumo_cmd():
    """Test command generation for both headless (sumo) and GUI (sumo-gui) modes."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    runner = SimulationRunner(
        cfg_path=cfg_path,
        tripinfo_output="tripinfo.xml",
        additional_args=["--no-step-log", "true"],
    )

    cmd_headless = runner.build_sumo_cmd(gui=False)
    assert any("sumo" in c.lower() for c in cmd_headless)
    assert "-c" in cmd_headless
    assert os.path.abspath(cfg_path) in cmd_headless
    assert "--tripinfo-output" in cmd_headless
    assert os.path.abspath("tripinfo.xml") in cmd_headless
    assert "--no-step-log" in cmd_headless

    cmd_gui = runner.build_sumo_cmd(gui=True)
    assert any("sumo-gui" in c.lower() for c in cmd_gui)


def test_simulation_runner_100_steps_cologne1():
    """Test short 100-step simulation execution on cologne1 scenario."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    info = inspect_scenario("cologne1", data_dir="data")
    controller = FixedTimeController(
        green_phases=info.get_green_phases(),
        green_time=30.0,
        yellow_time=3.0,
        tls_id=info.tls_ids[0],
    )

    runner = SimulationRunner(
        cfg_path=cfg_path,
        controller=controller,
        tls_id=info.tls_ids[0],
        step_duration=1.0,
    )

    # Run 100 steps
    result = runner.run(gui=False, max_steps=100)

    # Verify result structure and statistics
    assert isinstance(result, SimulationResult)
    assert result.total_steps == 100
    assert len(result.queue_history) == 100
    assert result.controller_step_count == 100
    assert result.mean_queue_length >= 0.0
    assert result.max_queue_length >= 0
    assert result.completed is True

    # Check that TraCI connection was safely closed
    assert not traci.isLoaded()


def test_simulation_runner_100_steps_ingolstadt1():
    """Test short 100-step simulation execution on ingolstadt1 scenario."""
    cfg_path = os.path.join("data", "ingolstadt1", "ingolstadt1.sumocfg")
    runner = SimulationRunner(cfg_path=cfg_path)

    result = runner.run(gui=False, max_steps=100)

    assert isinstance(result, SimulationResult)
    assert result.total_steps == 100
    assert len(result.queue_history) == 100
    assert result.controller_step_count == 100
    assert result.mean_queue_length >= 0.0
    assert result.max_queue_length >= 0
    assert result.completed is True
    assert not traci.isLoaded()


def test_simulation_runner_safe_cleanup_on_exception():
    """Ensure traci.close() is safely executed in try...finally when an exception occurs."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    runner = SimulationRunner(cfg_path=cfg_path)

    # Callback that raises an exception midway (step 15)
    def failing_callback(step_idx: int, state: str, queue: int):
        if step_idx >= 15:
            raise RuntimeError("Intentional error during simulation")

    with pytest.raises(RuntimeError, match="Intentional error during simulation"):
        runner.run(gui=False, max_steps=100, step_callback=failing_callback)

    # Verify that TraCI port was safely closed and not left dangling
    assert not traci.isLoaded()


def test_simulation_runner_with_tripinfo_output():
    """Test simulation produces valid tripinfo XML when requested."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    try:
        runner = SimulationRunner(
            cfg_path=cfg_path,
            tripinfo_output=tripinfo_path,
        )
        result = runner.run(gui=False, max_steps=50)

        assert result.total_steps == 50
        assert os.path.exists(tripinfo_path)
        assert os.path.getsize(tripinfo_path) > 0
    finally:
        if os.path.exists(tripinfo_path):
            os.remove(tripinfo_path)


def test_simulation_runner_context_manager():
    """Test SimulationRunner context manager cleans up TraCI automatically."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    runner = SimulationRunner(cfg_path=cfg_path)

    with runner:
        runner.start(gui=False)
        assert traci.isLoaded()
        # Advance 5 steps manually
        for _ in range(5):
            runner.step()

    # After exiting with block, traci must be closed
    assert not traci.isLoaded()
