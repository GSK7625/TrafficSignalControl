"""TraCI Simulation Runner module for SUMO with traffic signal control (Fixed Time & Max Pressure)."""

from dataclasses import dataclass, field
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import sumolib
import traci

from src.controllers import FixedTimeController, MaxPressureController, QLearningController
from src.data_inspector import inspect_scenario, ScenarioInfo


@dataclass
class SimulationResult:
    """Result data container for a simulation run."""
    total_steps: int
    sim_time_start: float
    sim_time_end: float
    queue_history: List[int]
    controller_step_count: int
    mean_queue_length: float
    max_queue_length: int
    tripinfo_path: Optional[str] = None
    completed: bool = True
    tls_id: Optional[str] = None
    controller_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts result metrics to dictionary format."""
        d = {
            "total_steps": self.total_steps,
            "sim_time_start": self.sim_time_start,
            "sim_time_end": self.sim_time_end,
            "mean_queue_length": self.mean_queue_length,
            "max_queue_length": self.max_queue_length,
            "controller_step_count": self.controller_step_count,
            "tripinfo_path": self.tripinfo_path,
            "completed": self.completed,
            "tls_id": self.tls_id,
        }
        if self.controller_type is not None:
            d["controller_type"] = self.controller_type
        return d


class SimulationRunner:
    """Orchestrates SUMO simulation via TraCI with traffic signal control (Fixed Time or Max Pressure).

    Coordinates the simulation loop per second:
      1. Updates traffic signal phase via controller
      2. Measures queue length (halting vehicles v < 0.1 m/s) on incoming lanes
      3. Advances simulation step (traci.simulationStep())
    Safely handles TraCI connection closure in try...finally blocks.
    """

    def __init__(
        self,
        cfg_path: str,
        controller: Optional[Any] = None,
        tls_id: Optional[str] = None,
        step_duration: float = 1.0,
        tripinfo_output: Optional[str] = None,
        incoming_lanes: Optional[Sequence[str]] = None,
        additional_args: Optional[Sequence[str]] = None,
        controller_type: str = "fixed_time",
    ) -> None:
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"SUMO configuration file not found: {cfg_path}")

        if controller_type not in ("fixed_time", "max_pressure", "q_learning"):
            raise ValueError(f"Unsupported controller_type: {controller_type}. Must be 'fixed_time', 'max_pressure', or 'q_learning'.")

        self.cfg_path = os.path.abspath(cfg_path)
        self.step_duration = float(step_duration)
        self.tripinfo_output = os.path.abspath(tripinfo_output) if tripinfo_output else None
        self.additional_args = list(additional_args) if additional_args else []
        self._is_running = False

        # Load scenario metadata if controller or tls_id or lanes are omitted
        scenario_info: Optional[ScenarioInfo] = None
        if controller is None or tls_id is None or incoming_lanes is None:
            try:
                scenario_info = inspect_scenario(
                    scenario_name=os.path.splitext(os.path.basename(cfg_path))[0],
                    cfg_path=self.cfg_path,
                )
            except Exception:
                scenario_info = None

        # Resolve TLS ID
        if tls_id:
            self.tls_id = tls_id
        elif controller and getattr(controller, "tls_id", None):
            self.tls_id = controller.tls_id
        elif scenario_info and scenario_info.tls_ids:
            self.tls_id = scenario_info.tls_ids[0]
        else:
            self.tls_id = ""

        # Resolve controller and controller_type
        if controller is not None:
            self.controller = controller
            if isinstance(controller, MaxPressureController):
                self.controller_type = "max_pressure"
            elif isinstance(controller, FixedTimeController):
                self.controller_type = "fixed_time"
            else:
                self.controller_type = controller_type
        else:
            self.controller_type = controller_type
            if scenario_info is None:
                raise ValueError(f"Cannot initialize {self.controller_type} controller without scenario metadata.")
            green_phases = scenario_info.get_green_phases(self.tls_id)
            if not green_phases:
                raise ValueError(f"No green phases found for TLS '{self.tls_id}'.")

            if self.controller_type == "max_pressure":
                movements = scenario_info.get_movements(self.tls_id)
                self.controller = MaxPressureController(
                    green_phases=green_phases,
                    movements=movements,
                    min_green=10.0,
                    max_green=60.0,
                    yellow_time=3.0,
                    tls_id=self.tls_id,
                )
            else:
                self.controller = FixedTimeController(
                    green_phases=green_phases,
                    green_time=30.0,
                    yellow_time=3.0,
                    tls_id=self.tls_id,
                )

        # Resolve incoming lanes
        if incoming_lanes is not None:
            self.incoming_lanes: List[str] = sorted(list(set(incoming_lanes)))
        elif scenario_info and self.tls_id in scenario_info.controlled_lanes_per_tls:
            self.incoming_lanes = sorted(list(set(scenario_info.controlled_lanes_per_tls[self.tls_id])))
        else:
            self.incoming_lanes = []

    def build_sumo_cmd(self, gui: bool = False) -> List[str]:
        """Constructs the SUMO startup command arguments.

        Args:
            gui: True to use 'sumo-gui' graphical interface, False for headless 'sumo'.

        Returns:
            List of command line arguments for traci.start().
        """
        binary_name = "sumo-gui" if gui else "sumo"
        try:
            sumo_bin = sumolib.checkBinary(binary_name)
        except Exception:
            sumo_bin = binary_name

        cmd = [
            sumo_bin,
            "-c",
            self.cfg_path,
            "--no-warnings",
            "true",
        ]

        if self.tripinfo_output:
            cmd.extend(["--tripinfo-output", self.tripinfo_output])

        # Append additional arguments (avoiding exact duplicates of flags)
        if self.additional_args:
            cmd.extend(self.additional_args)

        return cmd

    def start(self, gui: bool = False) -> None:
        """Starts the SUMO process and connects TraCI.

        Args:
            gui: If True, launches sumo-gui. Otherwise launches headless sumo.
        """
        # Ensure any dangling connection is closed before starting new one
        self.close()

        cmd = self.build_sumo_cmd(gui=gui)
        traci.start(cmd)
        self._is_running = True

        # If incoming lanes were not known ahead of time, query them dynamically
        if not self.incoming_lanes and self.tls_id:
            try:
                lanes = traci.trafficlight.getControlledLanes(self.tls_id)
                self.incoming_lanes = sorted(list(set(lanes)))
            except Exception:
                pass

    def measure_queue(self) -> int:
        """Measures total number of halting vehicles (speed < 0.1 m/s) on incoming lanes.

        Returns:
            Total queue count across all monitored incoming lanes.
        """
        if not self.incoming_lanes:
            return 0
        total_halted = 0
        for lane_id in self.incoming_lanes:
            try:
                total_halted += traci.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                pass
        return total_halted

    def step(self) -> Tuple[str, int]:
        """Executes a single coordination step:

        1. Updates traffic signal phase via FixedTimeController
        2. Measures queue length on incoming lanes
        3. Advances simulation step (traci.simulationStep())

        Returns:
            Tuple[str, int]: (signal_state, queue_length)
        """
        # 1. Update traffic signal state
        state, _ = self.controller.step(self.step_duration)
        if self.tls_id:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, state)

        # 2. Measure queue length
        queue = self.measure_queue()

        # 3. Next simulation step
        traci.simulationStep()

        return state, queue

    def close(self) -> None:
        """Safely closes TraCI connection if currently loaded."""
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        finally:
            self._is_running = False

    def __enter__(self) -> "SimulationRunner":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit ensures TraCI is closed."""
        self.close()

    def run(
        self,
        gui: bool = False,
        max_steps: int = 3600,
        step_callback: Optional[Callable[[int, str, int], None]] = None,
    ) -> SimulationResult:
        """Runs the simulation for up to max_steps with safe exception handling.

        Args:
            gui: True for GUI mode, False for headless.
            max_steps: Maximum number of simulation seconds/steps to run.
            step_callback: Optional callback invoked per step as (step_idx, state, queue).

        Returns:
            SimulationResult containing step counts and queue metrics.
        """
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")

        queue_history: List[int] = []
        sim_time_start: float = 0.0
        sim_time_end: float = 0.0
        completed = False

        self.controller.reset()

        try:
            self.start(gui=gui)
            sim_time_start = float(traci.simulation.getTime())

            for step_idx in range(max_steps):
                # 1. Cập nhật tín hiệu đèn Fixed Time
                state, _ = self.controller.step(self.step_duration)
                if self.tls_id:
                    traci.trafficlight.setRedYellowGreenState(self.tls_id, state)

                # 2. Đo hàng đợi
                current_queue = self.measure_queue()
                queue_history.append(current_queue)

                if step_callback is not None:
                    step_callback(step_idx, state, current_queue)

                # 3. Bước mô phỏng tiếp theo
                traci.simulationStep()

                # If no more vehicles exist and simulation can end early
                if traci.simulation.getMinExpectedNumber() <= 0 and step_idx > 10:
                    break

            sim_time_end = float(traci.simulation.getTime())
            completed = True
        finally:
            # Luôn đảm bảo đóng TraCI an toàn
            self.close()

        mean_queue = float(sum(queue_history) / len(queue_history)) if queue_history else 0.0
        max_queue = max(queue_history) if queue_history else 0

        return SimulationResult(
            total_steps=len(queue_history),
            sim_time_start=sim_time_start,
            sim_time_end=sim_time_end,
            queue_history=queue_history,
            controller_step_count=self.controller.step_count,
            mean_queue_length=mean_queue,
            max_queue_length=max_queue,
            tripinfo_path=self.tripinfo_output,
            completed=completed,
            tls_id=self.tls_id,
            controller_type=self.controller_type,
        )
