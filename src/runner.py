"""Simulation Runner orchestrating SUMO, TraCI, SafePhaseManager, and MetricsCollector."""

from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import traci
except ImportError:
    traci = None

from src.config import SimulationConfig, get_scenario_config
from src.data_inspector import PhaseInfo, ScenarioInfo, inspect_scenario
from src.metrics import MetricsCollector, StepMetrics
from src.phase_manager import SafePhaseManager


def extract_green_phases(scenario_name: str, tl_id: str, phases: List[PhaseInfo]) -> List[str]:
    """Extracts valid selectable green phases for a given traffic light.

    Filters out intermediate yellow clearance phases while preserving all genuine green configurations.
    """
    green_states = []
    for p in phases:
        has_green = any(c in ("G", "g") for c in p.state)
        # Identify intermediate yellow clearance transitions (typically 3-4s with yellow active)
        is_yellow_clearance = (
            "y" in p.state
            and p.duration <= 4.0
            and p.state.count("y") >= p.state.count("G")
        )
        if has_green and not is_yellow_clearance:
            green_states.append(p.state)

    if not green_states:
        # Fallback: take all phases that have any green signal
        green_states = [p.state for p in phases if any(c in ("G", "g") for c in p.state)]
    if not green_states:
        # Extreme fallback: take all defined phases
        green_states = [p.state for p in phases]

    return green_states


class SimulationRunner:
    """Manages the full lifecycle of a SUMO simulation using TraCI."""

    def __init__(self, config: SimulationConfig, control_mode: Optional[str] = None):
        self.config = config
        self.control_mode = control_mode if control_mode is not None else config.control_mode
        self.scenario_info: ScenarioInfo = inspect_scenario(
            config.dataset_name, data_dir=config.data_dir
        )

        # Aggregate all controlled incoming lanes across intersections
        self.all_controlled_lanes: List[str] = []
        for lanes in self.scenario_info.controlled_lanes_per_tls.values():
            self.all_controlled_lanes.extend(lanes)
        self.all_controlled_lanes = sorted(list(set(self.all_controlled_lanes)))

        self.has_run: bool = False
        self.reset()

    def reset(self) -> None:
        """Resets the runner to initial state, generating fresh run_id, metrics collector,

        and phase managers to ensure subsequent runs do not contaminate each other.
        """
        if getattr(self, "is_running", False):
            self.close()

        import uuid
        run_uuid = uuid.uuid4().hex[:6]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{run_uuid}"
        self.run_folder_name = f"{self.config.dataset_name}_{self.run_id}"

        # Initialize SafePhaseManager for each traffic light
        self.tls_managers: Dict[str, SafePhaseManager] = {}
        for tl_id, phases in self.scenario_info.phases_per_tls.items():
            green_states = extract_green_phases(self.config.dataset_name, tl_id, phases)
            self.tls_managers[tl_id] = SafePhaseManager(
                tls_id=tl_id,
                green_phases=green_states,
                yellow_duration=self.config.yellow_duration,
                all_red_duration=self.config.all_red_duration,
                min_green=self.config.min_green,
            )

        # Unique temporary tripinfo path if enabled
        self.tripinfo_path: Optional[str] = None
        if self.config.use_tripinfo:
            import tempfile
            self.tripinfo_path = os.path.join(
                tempfile.gettempdir(), f"tripinfo_{self.run_folder_name}.xml"
            )

        # Initialize fresh metrics collector
        self.metrics_collector = MetricsCollector(
            scenario_name=self.config.dataset_name,
            tripinfo_file=self.tripinfo_path,
        )

        self.current_step: int = 0
        self.sim_time: float = self.config.begin_time
        self.is_running: bool = False
        self._active_control_engaged: bool = (self.control_mode in ("cyclic", "agent"))
        self.total_teleports: int = 0
        self.total_collisions: int = 0
        self.unique_colliding_vehicles: set = set()
        self.total_collision_incidents: int = 0
        self.total_collision_steps: int = 0
        self.total_collision_vehicle_involvements: int = 0
        self.has_run: bool = False

    def build_sumo_cmd(self) -> List[str]:
        """Constructs the command arguments for starting SUMO via TraCI."""
        cmd = [
            self.config.sumo_binary,
            "-c",
            self.config.sumocfg_path,
            "-b",
            str(self.config.begin_time),
            "-e",
            str(self.config.end_time),
            "--step-length",
            str(self.config.step_length),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            "10000",
            "--time-to-teleport",
            "-1",
        ]

        if self.tripinfo_path:
            cmd.extend(["--tripinfo-output", self.tripinfo_path])

        if self.config.seed is not None:
            cmd.extend(["--seed", str(self.config.seed)])

        if self.config.gui:
            cmd.extend(["--start", "--quit-on-end"])

        return cmd

    def start(self) -> None:
        """Starts SUMO and establishes TraCI connection."""
        if self.is_running:
            return

        if self.has_run or self.current_step > 0 or len(self.metrics_collector.step_records) > 0:
            self.reset()

        if traci is None:
            raise RuntimeError("TraCI module is not installed or available in this environment")

        cmd = self.build_sumo_cmd()
        traci.start(cmd)
        self.is_running = True
        self.current_step = 0
        self.sim_time = self.config.begin_time

        # Apply initial green phases only if active control is engaged (not in original baseline mode)
        if self._active_control_engaged:
            for manager in self.tls_managers.values():
                manager.apply_to_traci(traci)

    def step(self, actions: Optional[Dict[str, int]] = None) -> StepMetrics:
        """Executes a single simulation step.

        Args:
            actions: Optional mapping of {tls_id: target_phase_idx} for phase change requests.
                     If an empty dict {} is passed, active control is engaged but current phases are held.
                     If None is passed, behavior depends on control_mode:
                       - 'original': native SUMO traffic light logic runs without TraCI override.
                       - 'cyclic': cycles to next green phase after 30 seconds.
                       - 'agent': maintains currently held phases.

        Returns:
            StepMetrics for the completed step.
        """
        if not self.is_running:
            raise RuntimeError("SimulationRunner is not running. Call start() first.")

        # 1. Process phase action requests or control modes
        if actions is not None:
            if self.control_mode == "original":
                raise ValueError(
                    "Cannot pass external actions when SimulationRunner is configured with control_mode='original'. "
                    "Use control_mode='agent' or control_mode='cyclic' to enable external signal control."
                )
            self._active_control_engaged = True
            for tls_id, target_phase in actions.items():
                if tls_id in self.tls_managers:
                    self.tls_managers[tls_id].request_phase(target_phase)
        elif self.control_mode == "cyclic":
            self._active_control_engaged = True
            # Default progression: cycle to next green phase after 30 seconds
            for manager in self.tls_managers.values():
                if (
                    not manager.is_in_transition()
                    and manager.time_in_current_phase() >= 30.0
                    and len(manager.green_phases) > 1
                ):
                    next_p = (manager.get_current_phase() + 1) % len(manager.green_phases)
                    manager.request_phase(next_p)

        # 2. Apply current signal strings to TraCI if active control is engaged
        if self._active_control_engaged:
            for manager in self.tls_managers.values():
                manager.apply_to_traci(traci)

        # 3. Advance SUMO simulation
        traci.simulationStep()
        self.current_step += 1
        self.sim_time = traci.simulation.getTime()

        # Accumulate teleports and collisions occurred in this step
        if traci is not None and self.is_running:
            try:
                new_teleports = traci.simulation.getStartingTeleportNumber()
                if isinstance(new_teleports, (int, float)):
                    self.total_teleports += int(new_teleports)
            except Exception as e:
                self.metrics_collector.record_error(
                    f"Step {self.current_step} getStartingTeleportNumber error: {e}"
                )

            step_has_collision = False
            collisions_list = None
            collision_query_success = False
            try:
                collisions_list = traci.simulation.getCollisions()
                if collisions_list:
                    step_has_collision = True
                    self.total_collision_incidents += len(collisions_list)
                    for col in collisions_list:
                        c_id = getattr(col, "collider", None)
                        v_id = getattr(col, "victim", None)
                        if c_id:
                            self.unique_colliding_vehicles.add(c_id)
                        if v_id:
                            self.unique_colliding_vehicles.add(v_id)
                collision_query_success = True
            except Exception:
                collisions_list = None

            try:
                colliding_vehs = traci.simulation.getCollidingVehiclesIDList()
                if colliding_vehs:
                    step_has_collision = True
                    self.total_collisions += len(colliding_vehs)
                    self.total_collision_vehicle_involvements += len(colliding_vehs)
                    self.unique_colliding_vehicles.update(colliding_vehs)
                    if collisions_list is None:
                        self.total_collision_incidents += max(1, len(colliding_vehs) // 2)
                collision_query_success = True
            except Exception:
                try:
                    coll_num = traci.simulation.getCollidingVehiclesNumber()
                    if isinstance(coll_num, (int, float)):
                        if coll_num > 0:
                            step_has_collision = True
                            self.total_collisions += int(coll_num)
                            self.total_collision_vehicle_involvements += int(coll_num)
                            if collisions_list is None:
                                self.total_collision_incidents += max(1, int(coll_num) // 2)
                        collision_query_success = True
                except Exception:
                    pass

            if not collision_query_success:
                self.metrics_collector.record_error(
                    f"Step {self.current_step} all collision getters failed to retrieve telemetry"
                )

            if step_has_collision:
                self.total_collision_steps += 1

        # 4. Advance phase manager internal timers after simulation step executed
        for manager in self.tls_managers.values():
            manager.update(step_duration=self.config.step_length)

        # 5. Record step metrics
        metrics = self.metrics_collector.record_step(
            step=self.current_step,
            sim_time=self.sim_time,
            controlled_lanes=self.all_controlled_lanes,
            traci_module=traci,
            step_length=self.config.step_length,
        )

        return metrics

    def run(self, num_steps: Optional[int] = None) -> Dict[str, Any]:
        """Runs the simulation until num_steps or scenario completion, then saves results.

        Args:
            num_steps: Optional maximum number of steps to execute.

        Returns:
            Dictionary containing the summary of the simulation run.
        """
        max_steps = num_steps if num_steps is not None else self.config.max_steps

        try:
            self.start()
            while True:
                # Check termination conditions
                if max_steps is not None and self.current_step >= max_steps:
                    break
                if self.sim_time >= self.config.end_time:
                    break

                self.step()

        finally:
            # Query unserved vehicles and network stats before closing SUMO
            running_vehs = 0
            pending_vehs = 0
            if traci is not None and self.is_running:
                try:
                    running_vehs = len(traci.vehicle.getIDList())
                except Exception as e:
                    self.metrics_collector.record_error(
                        f"End-of-run getIDList error: {e}"
                    )
                try:
                    pending_vehs = len(traci.simulation.getPendingVehicles())
                except Exception as e:
                    self.metrics_collector.record_error(
                        f"End-of-run getPendingVehicles error: {e}"
                    )

            metadata = {
                "control_mode": self.control_mode,
                "seed": self.config.seed,
                "step_length": self.config.step_length,
                "yellow_duration": self.config.yellow_duration,
                "all_red_duration": self.config.all_red_duration,
                "min_green": self.config.min_green,
                "running_vehicles_at_end": running_vehs,
                "pending_insertion_vehicles": pending_vehs,
                "teleport_count": self.total_teleports,
                "collision_count": self.total_collision_incidents,
                "collision_incidents": self.total_collision_incidents,
                "collision_steps": self.total_collision_steps,
                "collision_vehicle_involvements": self.total_collision_vehicle_involvements,
                "unique_colliding_vehicles_count": len(self.unique_colliding_vehicles),
            }
            self.metrics_collector.set_run_metadata(metadata)
            self.close()
            self.has_run = True

        # Export metrics into unique run folder
        run_folder = os.path.join(self.config.output_dir, self.run_folder_name)
        os.makedirs(run_folder, exist_ok=True)

        if self.tripinfo_path and os.path.isfile(self.tripinfo_path):
            import shutil
            final_tripinfo = os.path.join(run_folder, "tripinfo.xml")
            shutil.copy2(self.tripinfo_path, final_tripinfo)
            try:
                os.remove(self.tripinfo_path)
            except Exception:
                pass
            self.metrics_collector.tripinfo_file = final_tripinfo

        json_path, csv_path = self.metrics_collector.export_results(run_folder)

        summary = self.metrics_collector.get_summary()
        summary["output_json"] = json_path
        summary["output_csv"] = csv_path
        return summary

    def close(self) -> None:
        """Closes the TraCI connection safely and marks session as completed."""
        if self.is_running and traci is not None:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                self.is_running = False
                self.has_run = True
        elif not self.is_running and (self.current_step > 0 or len(self.metrics_collector.step_records) > 0):
            self.has_run = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
