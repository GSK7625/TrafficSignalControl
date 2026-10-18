"""Simulation Runner orchestrating SUMO, TraCI, SafePhaseManager, and MetricsCollector."""

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union

try:
    import traci
except ImportError:
    traci = None

from src.config import SimulationConfig
from src.controllers.base import ControlledLink, IntersectionObservation
from src.data_inspector import PhaseInfo, ScenarioInfo, inspect_scenario
from src.metrics import MetricsCollector, StepMetrics, build_vehicle_accounting
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
        # Fallback: if all phases contain yellow, keep phases where green signals outnumber yellow
        for p in phases:
            num_green = sum(1 for c in p.state if c in ("G", "g"))
            num_yellow = sum(1 for c in p.state if c in ("y", "Y"))
            if num_green > num_yellow and p.state not in green_states:
                green_states.append(p.state)

    if not green_states:
        raise ValueError(f"Dataset '{scenario_name}' TLS '{tl_id}' has no selectable pure-green phase (yellow/all-red are not actions)")

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
        self.target_tls_id: Optional[str] = self.config.target_tls_id
        self.controlled_tls_ids: Optional[List[str]] = (
            list(self.config.controlled_tls_ids) if self.config.controlled_tls_ids is not None else None
        )
        self.reset()

    def get_controlled_tls_ids(self) -> Set[str]:
        """Returns the set of actively controlled traffic light IDs."""
        return set(self._controlled_tls_ids)

    def set_controlled_tls_ids(self, tls_ids: Union[str, Sequence[str]]) -> None:
        """Sets the active traffic light IDs to control, persisting across simulation resets."""
        if isinstance(tls_ids, str):
            ids = [tls_ids]
        else:
            ids = list(tls_ids)
        self.config.controlled_tls_ids = ids
        if len(ids) == 1:
            self.config.target_tls_id = ids[0]
            self.target_tls_id = ids[0]
        self.controlled_tls_ids = ids
        self._controlled_tls_ids = set(ids)

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

        import tempfile
        self.sumo_log_path: str = os.path.join(
            tempfile.gettempdir(), f"sumo_{self.run_folder_name}.log"
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
            if self.config.controlled_tls_ids is not None:
                self._controlled_tls_ids = set(self.config.controlled_tls_ids)
            elif self.config.target_tls_id is not None:
                self._controlled_tls_ids = {self.config.target_tls_id}
            elif self.controller is not None and getattr(self.controller, "target_tls_id", None) is not None:
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
        self.safety_events: List[Dict[str, Any]] = []
        self.last_action_results: Dict[str, bool] = {}
        self.native_program_load_warnings: List[str] = []
        self._load_warning_capture_phase = "at_start"

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
            "false",
            "--log",
            self.sumo_log_path,
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

        # SUMO loads its native programs before any external state is applied.
        # Preserve load-time warning provenance; these are not custom transitions.
        if os.path.isfile(self.sumo_log_path):
            with open(self.sumo_log_path, encoding="utf-8", errors="replace") as log:
                self.native_program_load_warnings = [
                    line.strip() for line in log if "warning" in line.lower()
                ]

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

            pending_cnt: Optional[int] = 0
            if traci is not None:
                try:
                    pending_ids = traci.simulation.getPendingVehicles()
                    pending_cnt = len(pending_ids) if pending_ids is not None else 0
                except Exception as e:
                    self.metrics_collector.record_error(
                        f"Step {self.current_step} getPendingVehicles error: {e}"
                    )
                    pending_cnt = None

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
                     None/{} hold only while safety permits; max-green is enforced each step.
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

        self.last_action_results = {}
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
                    source_phase = mgr.get_current_phase()
                    accepted = mgr.request_phase(target_phase)
                    self.last_action_results[tls_id] = accepted
                    reason = ("accepted" if accepted else "clearance_locked" if mgr.is_in_transition()
                              else "max_green_hold_rejected" if mgr.is_max_green_exceeded() else "min_green")
                    self._record_safety_event(tls_id, source_phase, target_phase, reason,
                                              requested_action=target_phase, accepted=accepted, forced=False,
                                              accepted_action=target_phase if accepted else None, forced_action=None)
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
                    source = manager.get_current_phase()
                    accepted = manager.request_phase(next_p)
                    self._record_safety_event(manager.tls_id, source, next_p, "cyclic_30s_request",
                                              requested_action=next_p, accepted=accepted, forced=False,
                                              accepted_action=next_p if accepted else None, forced_action=None)

        # One owner: enforce at the boundary BEFORE the next simulation interval.
        # Completed green at its cap must not enter another green interval.
        # Clearance keeps its committed target. Single-phase and None are exempt.
        for tls_id in sorted(self._controlled_tls_ids):
            mgr = self.tls_managers[tls_id]
            if mgr.is_max_green_exceeded() and len(mgr.green_phases) > 1:
                source_phase = mgr.get_current_phase()
                forced_p = (source_phase + 1) % len(mgr.green_phases)
                accepted = mgr.request_phase(forced_p)
                self._record_safety_event(tls_id, source_phase, forced_p, "max_green_cyclic_fallback",
                                          requested_action=None, accepted=accepted, forced=True,
                                          accepted_action=forced_p if accepted else None, forced_action=forced_p)

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

    def _record_safety_event(self, tls_id, source, target, reason, **details):
        self.safety_events.append({
            "tls_id": tls_id, "sim_time": self.sim_time,
            "source": source, "target": target, "reason": reason,
            **details,
        })

    def safety_metadata(self) -> Dict[str, Any]:
        """Protocol and warning provenance shared by CLI and RL exports."""
        return {
            "max_green": self.config.max_green,
            "safety_events": self.safety_events,
            "native_program_load_warnings": self.native_program_load_warnings,
            "native_program_load_warning_provenance": {
                "source": "SUMO native program load before external overrides",
                "sim_time": self.config.begin_time,
                "log_path": self.sumo_log_path,
                "capture_phase": self._load_warning_capture_phase,
                "buffered_recovery_rule": "Only native (non-online) Missing yellow phase in tlLogic program-validation diagnostics; other runtime warnings are not attributed to load",
            },
            "max_green_exemptions": {
                tls: "disabled" if mgr.max_green is None else "single_selectable_phase"
                for tls, mgr in self.tls_managers.items()
                if tls in self._controlled_tls_ids and (mgr.max_green is None or len(mgr.green_phases) == 1)
            },
        }

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

            accounting_meta = build_vehicle_accounting(
                scenario_name=getattr(self.config, "dataset_name", getattr(self.config, "scenario_name", "")),
                window_begin=self.config.begin_time,
                window_end=self.sim_time,
                step_length=self.config.step_length,
                total_completed=self.metrics_collector._cumulative_completed,
                running_vehs=running_vehs,
                pending_vehs=pending_vehs,
                total_teleports=self.total_teleports,
                total_collisions=self.total_collision_incidents,
                scenario_info=self.scenario_info,
                population=self.metrics_collector.population_snapshot(),
            )

            metadata = {
                "control_mode": self.control_mode,
                "seed": self.config.seed,
                "step_length": self.config.step_length,
                "yellow_duration": self.config.yellow_duration,
                "all_red_duration": self.config.all_red_duration,
                "min_green": self.config.min_green,
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
            metadata.update(accounting_meta)
            metadata.update(self.safety_metadata())
            if self.controller is not None:
                metadata["controller_name"] = self.controller.name
                metadata["controller_config"] = self.controller.get_config()

            # Close TraCI and terminate SUMO process so all log buffers are flushed to disk
            self.close()
            metadata.update(self.safety_metadata())
            self.has_run = True

            sumo_warnings = []
            if hasattr(self, "sumo_log_path") and self.sumo_log_path and os.path.isfile(self.sumo_log_path):
                try:
                    with open(self.sumo_log_path, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            sline = line.strip()
                            if sline.startswith("Warning:") or "warning" in sline.lower():
                                sumo_warnings.append(sline)
                except Exception:
                    pass

            metadata["sumo_warnings_count"] = len(sumo_warnings)
            metadata["sumo_warnings"] = sumo_warnings[:50]
            metadata["safety_protocol"] = (
                "native_sumo (uncontrolled external program; does not guarantee project clearance intervals)"
                if self.control_mode in ("sumo", "original")
                else "custom_safe_phase_manager (enforces yellow clearance, min_green, all_red, per-step max_green)"
            )

            self.metrics_collector.set_run_metadata(metadata)

        # Export metrics into unique run folder
        run_folder = os.path.join(self.config.output_dir, self.run_folder_name)
        os.makedirs(run_folder, exist_ok=True)

        if hasattr(self, "sumo_log_path") and self.sumo_log_path and os.path.isfile(self.sumo_log_path):
            import shutil
            final_sumo_log = os.path.join(run_folder, "sumo_warnings.log")
            try:
                shutil.copy2(self.sumo_log_path, final_sumo_log)
                os.remove(self.sumo_log_path)
            except Exception:
                pass

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
        summary["output_dir"] = run_folder
        summary["sumo_warnings_count"] = metadata.get("sumo_warnings_count", 0)
        summary["safety_protocol"] = metadata.get("safety_protocol", "unknown")
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
                self._recover_buffered_native_warnings()
        elif not self.is_running and (self.current_step > 0 or len(self.metrics_collector.step_records) > 0):
            self.has_run = True

    def _recover_buffered_native_warnings(self) -> None:
        """Recover narrowly identified native validation diagnostics after log flush.

        SUMO may buffer its log until close. Native missing-yellow diagnostics
        are emitted during program validation at load, confirmed by zero-step
        load-only probes. We never reload native programs during a run. Exclude
        the TraCI 'online' program and all generic runtime warnings.
        """
        self._load_warning_capture_phase = "after_close"
        try:
            if os.path.isfile(self.sumo_log_path):
                with open(self.sumo_log_path, encoding="utf-8", errors="replace") as log:
                    for line in log:
                        if re.search(r"Missing yellow phase in tlLogic '[^']+',? program '(?!online')[^']+' for tl-index \d+ when switching to phase \d+", line):
                            warning = line.strip()
                            if warning not in self.native_program_load_warnings:
                                self.native_program_load_warnings.append(warning)
        except OSError as exc:
            self.metrics_collector.record_error(f"Native program warning provenance read failed: {exc}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
