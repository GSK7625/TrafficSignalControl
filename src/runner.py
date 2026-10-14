"""Simulation Runner orchestrating SUMO, TraCI, SafePhaseManager, and MetricsCollector."""

from datetime import datetime
import os
from typing import Any, Dict, List, Optional

try:
    import traci
except ImportError:
    traci = None

from src.config import SimulationConfig
from src.controllers.base import ControlledLink, IntersectionObservation
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
        has_yellow = any(c in ("y", "Y") for c in p.state)
        # Any phase displaying yellow is a clearance transition and must NOT be an action
        if has_green and not has_yellow:
            if p.state not in green_states:
                green_states.append(p.state)

    if not green_states:
        # Fallback: take phases that have any green and strictly fewer yellow than green
        green_states = [
            p.state for p in phases
            if any(c in ("G", "g") for c in p.state)
            and p.state.count("y") < p.state.count("G")
        ]
    if not green_states:
        # Extreme fallback: take all phases that have any green signal
        green_states = [p.state for p in phases if any(c in ("G", "g") for c in p.state)]
    if not green_states:
        # Final fallback
        green_states = [p.state for p in phases]

    return green_states


class SimulationRunner:
    """Manages the full lifecycle of a SUMO simulation using TraCI."""

    def __init__(
        self,
        config: SimulationConfig,
        control_mode: Optional[str] = None,
        controller: Optional[Any] = None,
    ):
        self.config = config
        self.controller = controller
        if controller is not None:
            self.control_mode = "agent"
        else:
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
        self.tls_controlled_links: Dict[str, List[ControlledLink]] = {}
        self.reset()

    def reset(self) -> None:
        """Resets the runner to initial state, generating fresh run_id, metrics collector,

        and phase managers to ensure subsequent runs do not contaminate each other.
        """
        if getattr(self, "is_running", False):
            self.close()

        if self.controller is not None:
            self.controller.reset()

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
                max_green=getattr(self.config, "max_green", 60.0),
            )

        # Validate controller compatibility against safety constraints and scenario phases
        if hasattr(self.controller, "validate_compatibility"):
            self._validate_controller_compatibility(self.controller)

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
        self._controlled_tls_ids: set = set()
        if self.control_mode == "cyclic":
            self._controlled_tls_ids = set(self.tls_managers.keys())
        elif self.control_mode == "agent":
            if self.controller is not None and getattr(self.controller, "target_tls_id", None) is not None:
                self._controlled_tls_ids = {self.controller.target_tls_id}
            elif self.controller is not None:
                self._controlled_tls_ids = set(self.tls_managers.keys())
            else:
                if len(self.tls_managers) == 1:
                    self._controlled_tls_ids = set(self.tls_managers.keys())
                else:
                    self._controlled_tls_ids = set()
        self.decision_interval: float = float(self.config.decision_interval)
        self._last_selected_actions: Optional[Dict[str, int]] = None
        self._steps_since_last_decision: int = 0
        self.total_teleports: int = 0
        self.total_collisions: int = 0
        self.unique_colliding_vehicles: set = set()
        self.total_collision_incidents: int = 0
        self.total_collision_steps: int = 0
        self.total_collision_vehicle_involvements: int = 0
        self.has_run: bool = False

    def _validate_controller_compatibility(self, controller: Any) -> None:
        """Validates controller configuration against runner safety constraints and scenario phases."""
        num_phases_per_tls = {
            tl_id: len(mgr.green_phases) for tl_id, mgr in self.tls_managers.items()
        }
        max_green = getattr(self.config, "max_green", 60.0)
        min_green = getattr(self.config, "min_green", 5.0)

        errors = controller.validate_compatibility(
            min_green=min_green,
            max_green=max_green,
            num_phases_per_tls=num_phases_per_tls,
        )
        if errors:
            ctrl_name = getattr(controller, "name", "Controller")
            raise ValueError(
                f"Incompatible controller configuration for {ctrl_name}: {'; '.join(errors)}"
            )

    def set_controller(self, controller: Any) -> None:
        """Attaches and validates a new controller for the simulation runner."""
        self.controller = controller
        self.control_mode = "agent"
        self._active_control_engaged = True
        if hasattr(self.controller, "validate_compatibility"):
            self._validate_controller_compatibility(self.controller)
        if getattr(self.controller, "target_tls_id", None) is not None:
            self._controlled_tls_ids = {self.controller.target_tls_id}
        else:
            self._controlled_tls_ids = set(self.tls_managers.keys())
        self.controller.reset()

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
            "--no-warnings",
            "true",
        ]

        if self.tripinfo_path:
            cmd.extend(["--tripinfo-output", self.tripinfo_path])

        if self.config.seed is not None:
            cmd.extend(["--seed", str(self.config.seed)])

        if self.config.gui:
            cmd.extend(["--start", "--quit-on-end"])
            if getattr(self.config, "gui_delay", 0.0) > 0:
                cmd.extend(["--delay", str(self.config.gui_delay)])

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

        # Query controlled links from TraCI for each traffic light
        for tl_id in self.tls_managers:
            try:
                raw_links = traci.trafficlight.getControlledLinks(tl_id)
                links_list = []
                for s_idx, conn_tuple in enumerate(raw_links):
                    for conn in conn_tuple:
                        from_l = conn[0] if len(conn) > 0 else ""
                        to_l = conn[1] if len(conn) > 1 else ""
                        via_l = conn[2] if len(conn) > 2 else None
                        links_list.append(
                            ControlledLink(
                                signal_index=s_idx,
                                from_lane=from_l,
                                to_lane=to_l,
                                via_lane=via_l,
                            )
                        )
                self.tls_controlled_links[tl_id] = links_list
            except Exception as e:
                self.tls_controlled_links[tl_id] = []
                self.metrics_collector.record_error(
                    f"Traffic light {tl_id} getControlledLinks query error: {e}"
                )

        # Apply initial green phases only if active control is engaged for the intersection
        if self._active_control_engaged:
            for tls_id in self._controlled_tls_ids:
                if tls_id in self.tls_managers:
                    self.tls_managers[tls_id].apply_to_traci(traci)
        else:
            for tls_id, manager in self.tls_managers.items():
                if tls_id not in self._controlled_tls_ids:
                    try:
                        curr_phase = traci.trafficlight.getPhase(tls_id)
                        curr_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                        manager.sync_external_phase(
                            phase_index=int(curr_phase) if isinstance(curr_phase, (int, float)) else None,
                            signal_state=curr_state if isinstance(curr_state, str) else None,
                            step_duration=0.0,
                        )
                    except Exception:
                        pass

    def get_observation(self) -> Dict[str, IntersectionObservation]:
        """Extracts current traffic state observation for all controlled intersections."""
        obs = {}
        for tl_id, mgr in self.tls_managers.items():
            links = self.tls_controlled_links.get(tl_id, [])
            in_lanes = set(l.from_lane for l in links if l.from_lane)
            out_lanes = set(l.to_lane for l in links if l.to_lane)

            in_queues: Dict[str, float] = {}
            in_vehs: Dict[str, float] = {}
            for lane_id in in_lanes:
                try:
                    in_queues[lane_id] = float(traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception as e:
                    in_queues[lane_id] = 0.0
                    self.metrics_collector.record_error(
                        f"Step {self.current_step} incoming lane {lane_id} getLastStepHaltingNumber error: {e}"
                    )
                try:
                    in_vehs[lane_id] = float(traci.lane.getLastStepVehicleNumber(lane_id))
                except Exception as e:
                    in_vehs[lane_id] = 0.0
                    self.metrics_collector.record_error(
                        f"Step {self.current_step} incoming lane {lane_id} getLastStepVehicleNumber error: {e}"
                    )

            out_queues: Dict[str, float] = {}
            out_vehs: Dict[str, float] = {}
            for lane_id in out_lanes:
                try:
                    out_queues[lane_id] = float(traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception as e:
                    out_queues[lane_id] = 0.0
                    self.metrics_collector.record_error(
                        f"Step {self.current_step} outgoing lane {lane_id} getLastStepHaltingNumber error: {e}"
                    )
                try:
                    out_vehs[lane_id] = float(traci.lane.getLastStepVehicleNumber(lane_id))
                except Exception as e:
                    out_vehs[lane_id] = 0.0
                    self.metrics_collector.record_error(
                        f"Step {self.current_step} outgoing lane {lane_id} getLastStepVehicleNumber error: {e}"
                    )

            pending_cnt = 0
            if traci is not None:
                try:
                    pending_cnt = int(traci.simulation.getPendingVehiclesNumber())
                except Exception:
                    pending_cnt = 0

            obs[tl_id] = IntersectionObservation(
                tls_id=tl_id,
                current_phase=mgr.get_current_phase(),
                time_in_phase=mgr.time_in_current_phase(),
                is_in_transition=mgr.is_in_transition(),
                green_phases=list(mgr.green_phases),
                controlled_links=links,
                in_lane_queues=in_queues,
                out_lane_queues=out_queues,
                in_lane_vehicles=in_vehs,
                out_lane_vehicles=out_vehs,
                sim_time=self.sim_time,
                step=self.current_step,
                pending_vehicles=pending_cnt,
            )
        return obs

    def step(self, actions: Optional[Dict[str, int]] = None) -> StepMetrics:
        """Executes a single simulation step.

        Args:
            actions: Optional mapping of {tls_id: target_phase_idx} for phase change requests.
                     If an empty dict {} is passed, active control is engaged but current phases are held.
                     If None is passed, behavior depends on control_mode:
                       - 'original': native SUMO traffic light logic runs without TraCI override.
                       - 'cyclic': cycles to next green phase after 30 seconds.
                       - 'agent': maintains currently held phases (or queries controller if registered).

        Returns:
            StepMetrics for the completed step.
        """
        if not self.is_running:
            raise RuntimeError("SimulationRunner is not running. Call start() first.")

        # 1. Process phase action requests or control modes
        if actions is None and self.controller is not None:
            decision_steps = max(1, int(round(self.decision_interval / self.config.step_length)))
            if self._last_selected_actions is None or self._steps_since_last_decision >= decision_steps:
                obs = self.get_observation()
                actions = self.controller.select_action(obs)
                self._last_selected_actions = actions
                self._steps_since_last_decision = 1
            else:
                actions = self._last_selected_actions
                self._steps_since_last_decision += 1
        elif actions is not None:
            self._last_selected_actions = actions
            self._steps_since_last_decision = 1

        if actions is not None:
            if self.control_mode == "original":
                raise ValueError(
                    "Cannot pass external actions when SimulationRunner is configured with control_mode='original'. "
                    "Use control_mode='agent' or control_mode='cyclic' to enable external signal control."
                )
            self._active_control_engaged = True
            for tls_id, target_phase in actions.items():
                if tls_id in self.tls_managers:
                    mgr = self.tls_managers[tls_id]
                    accepted = mgr.request_phase(target_phase)
                    if (
                        not accepted
                        and mgr.is_max_green_exceeded()
                        and not mgr.is_in_transition()
                        and len(mgr.green_phases) > 1
                    ):
                        forced_p = (mgr.get_current_phase() + 1) % len(mgr.green_phases)
                        mgr.request_phase(forced_p)
                    self._controlled_tls_ids.add(tls_id)
        elif self.control_mode == "cyclic":
            self._active_control_engaged = True
            self._controlled_tls_ids = set(self.tls_managers.keys())
            # Default progression: cycle to next green phase after 30 seconds
            for manager in self.tls_managers.values():
                if (
                    not manager.is_in_transition()
                    and manager.time_in_current_phase() >= 30.0
                    and len(manager.green_phases) > 1
                ):
                    next_p = (manager.get_current_phase() + 1) % len(manager.green_phases)
                    manager.request_phase(next_p)

        # 2. Apply current signal strings to TraCI for actively controlled intersections
        if self._active_control_engaged:
            for tls_id in self._controlled_tls_ids:
                if tls_id in self.tls_managers:
                    self.tls_managers[tls_id].apply_to_traci(traci)

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

        # 4. Advance phase manager internal timers or sync with SUMO for un-controlled TLS
        for tls_id, manager in self.tls_managers.items():
            if tls_id in self._controlled_tls_ids:
                manager.update(step_duration=self.config.step_length)
            else:
                # Intersections controlled by native SUMO (or control_mode == "original")
                synced = False
                if traci is not None and self.is_running:
                    try:
                        curr_phase = traci.trafficlight.getPhase(tls_id)
                        curr_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                        manager.sync_external_phase(
                            phase_index=int(curr_phase) if isinstance(curr_phase, (int, float)) else None,
                            signal_state=curr_state if isinstance(curr_state, str) else None,
                            step_duration=self.config.step_length,
                        )
                        synced = True
                    except Exception:
                        pass
                if not synced:
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

    def run(
        self,
        num_steps: Optional[int] = None,
        controller: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs the simulation until num_steps or scenario completion, then saves results.

        Args:
            num_steps: Optional maximum number of steps to execute.
            controller: Optional BaseTrafficController instance governing signal decisions.

        Returns:
            Dictionary containing the summary of the simulation run.
        """
        if controller is not None:
            self.set_controller(controller)

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
                "phase_change_count": sum(mgr.phase_switch_count for mgr in self.tls_managers.values()),
                "total_transition_time": round(sum(mgr.total_transition_time for mgr in self.tls_managers.values()), 2),
                "actual_yellow_time": round(sum(mgr.actual_yellow_time for mgr in self.tls_managers.values()), 2),
                "phase_durations_per_tls": {tl_id: mgr.phase_durations for tl_id, mgr in self.tls_managers.items()},
            }
            if self.controller is not None:
                metadata["controller_name"] = self.controller.name
                metadata["controller_config"] = self.controller.get_config()

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
