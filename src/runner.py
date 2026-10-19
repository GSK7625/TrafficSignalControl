"""Simulation Runner orchestrating SUMO, TraCI, SafePhaseManager, and MetricsCollector."""

from datetime import datetime
import os
import shutil
import tempfile
import copy
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union
import uuid

try:
    import traci
except ImportError:
    traci = None

try:
    import libsumo
except ImportError:
    libsumo = None

from src.config import SimulationConfig
from src.controllers.base import ControlledLink, IntersectionObservation
from src.data_inspector import PhaseInfo, ScenarioInfo
from src.metrics import MetricsCollector, StepMetrics, build_vehicle_accounting
from src.phase_manager import SafePhaseManager


def extract_green_phases(scenario_name: str, tl_id: str, phases: List[PhaseInfo]) -> List[str]:
    """Extracts selectable green phases for a traffic light (filtering out yellow clearance)."""
    greens = []
    for p in phases:
        has_green = any(c in "Gg" for c in p.state)
        has_yellow = any(c in "yY" for c in p.state)
        if has_green and not has_yellow and p.state not in greens:
            greens.append(p.state)

    if not greens:
        for p in phases:
            num_g = sum(1 for c in p.state if c in "Gg")
            num_y = sum(1 for c in p.state if c in "yY")
            if num_g > num_y and p.state not in greens:
                greens.append(p.state)

    if not greens:
        raise ValueError(f"No selectable green phases for TLS {tl_id}")
    return greens


class SimulationRunner:
    """Manages the full lifecycle of a SUMO simulation using TraCI or libsumo."""

    def __init__(
        self,
        config: SimulationConfig,
        control_mode: Optional[str] = None,
        controller: Optional[Any] = None,
    ):
        self.config = config
        self.controller = controller
        self.control_mode = "agent" if controller is not None else (control_mode or config.control_mode)
        if self.control_mode not in ("original", "agent"):
            raise ValueError(f"Unknown control_mode: {self.control_mode}; use FixedTimeController for cyclic schedules")

        self.scenario_info: ScenarioInfo = config.scenario_info
        self.all_controlled_lanes: List[str] = sorted({
            lane for lanes in self.scenario_info.controlled_lanes_per_tls.values() for lane in lanes
        })

        self.target_tls_id: Optional[str] = self.config.target_tls_id
        self.controlled_tls_ids: Optional[List[str]] = (
            list(self.config.controlled_tls_ids) if self.config.controlled_tls_ids else None
        )
        self.reset()

    def _resolve_sim_interface(self):
        if self.config.gui:
            if traci is None:
                raise RuntimeError("TraCI is required for GUI mode")
            return traci
        if self.config.use_libsumo and libsumo is not None:
            return libsumo
        if traci is not None:
            return traci
        raise RuntimeError("Neither libsumo nor TraCI is installed or available")

    def reset(self) -> None:
        """Resets the runner to initial state for a fresh simulation run."""
        if getattr(self, "is_running", False):
            self.close()

        self.traci = self._resolve_sim_interface()
        self.is_libsumo: bool = (getattr(self.traci, "__name__", "") == "libsumo")

        if self.controller is not None:
            self.controller.reset()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
        self.run_folder_name = f"{self.config.dataset_name}_{self.run_id}"

        # Setup SafePhaseManagers
        self.tls_managers: Dict[str, SafePhaseManager] = {}
        for tl_id, phases in self.scenario_info.phases_per_tls.items():
            greens = extract_green_phases(self.config.dataset_name, tl_id, phases)
            self.tls_managers[tl_id] = SafePhaseManager(
                tls_id=tl_id,
                green_phases=greens,
                yellow_duration=self.config.yellow_duration,
                all_red_duration=self.config.all_red_duration,
                min_green=self.config.min_green,
                max_green=self.config.max_green,
            )

        self.tripinfo_path: Optional[str] = (
            os.path.join(tempfile.gettempdir(), f"tripinfo_{self.run_folder_name}.xml")
            if self.config.use_tripinfo else None
        )
        self.sumo_log_path: str = os.path.join(
            tempfile.gettempdir(), f"sumo_{self.run_folder_name}.log"
        )

        self.metrics_collector = MetricsCollector(
            scenario_name=self.config.dataset_name,
            tripinfo_file=self.tripinfo_path,
        )

        self.current_step: int = 0
        self.sim_time: float = float(self.config.begin_time)
        self.is_running: bool = False
        self.has_run: bool = False
        self._final_summary = None
        self.tls_controlled_links: Dict[str, List[ControlledLink]] = {}

        if self.control_mode == "original":
            self._controlled_tls_ids = set()
        elif self.controlled_tls_ids:
            self._controlled_tls_ids = set(self.controlled_tls_ids)
        elif self.target_tls_id:
            self._controlled_tls_ids = {self.target_tls_id}
        elif self.controller and getattr(self.controller, "target_tls_id", None):
            self._controlled_tls_ids = {self.controller.target_tls_id}
        else:
            self._controlled_tls_ids = set(self.tls_managers.keys())

        if self._controlled_tls_ids - self.tls_managers.keys():
            raise ValueError("Controlled TLS not found in scenario")
        if self.controller is not None:
            self._validate_controller_compatibility(self.controller)

        self.decision_interval: float = float(self.config.decision_interval)
        self._last_selected_actions: Optional[Dict[str, int]] = None
        self._steps_since_last_decision: int = 0

        self.total_teleports: int = 0
        self.total_collision_incidents: int = 0
        self.total_collision_steps: int = 0
        self.total_collision_vehicle_involvements: int = 0
        self.unique_colliding_vehicles: Set[str] = set()

    def _validate_controller_compatibility(self, controller: Any) -> None:
        if hasattr(controller, "validate_compatibility"):
            num_phases = {tl: len(self.tls_managers[tl].green_phases) for tl in self._controlled_tls_ids}
            errors = controller.validate_compatibility(
                min_green=self.config.min_green,
                max_green=self.config.max_green,
                num_phases_per_tls=num_phases,
            )
            if errors:
                raise ValueError(f"Incompatible controller: {'; '.join(errors)}")

    def set_controller(self, controller: Any) -> None:
        self.controller = controller
        self.control_mode = "agent"
        target = getattr(controller, "target_tls_id", None)
        ids = self.config.controlled_tls_ids or ([self.config.target_tls_id] if self.config.target_tls_id else ([target] if target else list(self.tls_managers)))
        self.set_controlled_tls_ids(ids)
        self._validate_controller_compatibility(controller)
        self.controller.reset()

    def set_controlled_tls_ids(self, tls_ids: Union[str, Sequence[str]]) -> None:
        ids = [tls_ids] if isinstance(tls_ids, str) else list(tls_ids)
        if set(ids) - self.tls_managers.keys():
            raise ValueError(f"Unknown controlled TLS IDs: {set(ids) - self.tls_managers.keys()}")
        self.config.controlled_tls_ids = ids
        if len(ids) == 1:
            self.config.target_tls_id = ids[0]
            self.target_tls_id = ids[0]
        self.controlled_tls_ids = ids
        self._controlled_tls_ids = set(ids)

    def get_controlled_tls_ids(self) -> Set[str]:
        return set(self._controlled_tls_ids)

    def build_sumo_cmd(self) -> List[str]:
        cmd = [
            self.config.sumo_binary,
            "-c", self.config.sumocfg_path,
            "-b", str(self.config.begin_time),
            "-e", str(self.config.end_time),
            "--step-length", str(self.config.step_length),
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--no-warnings", "true",
            "--log", self.sumo_log_path,
            "--error-log", self.sumo_log_path,
        ]
        if self.tripinfo_path:
            cmd.extend(["--tripinfo-output", self.tripinfo_path])
        if self.config.seed is not None:
            cmd.extend(["--seed", str(self.config.seed)])
        if self.config.gui:
            cmd.extend(["--start", "--quit-on-end"])
            if self.config.gui_delay > 0:
                cmd.extend(["--delay", str(self.config.gui_delay)])
        return cmd

    def start(self) -> None:
        if self.is_running:
            return
        if self.has_run or self.current_step > 0:
            self.reset()

        if self.traci is None:
            raise RuntimeError("Simulation interface (traci/libsumo) not available")

        self.traci.start(self.build_sumo_cmd())
        self.is_running = True
        self.current_step = 0
        self.sim_time = self.config.begin_time

        for tl_id in self.tls_managers:
            raw = self.traci.trafficlight.getControlledLinks(tl_id)
            self.tls_controlled_links[tl_id] = [
                ControlledLink(s_idx, c[0], c[1], c[2] if len(c) > 2 else None)
                for s_idx, conns in enumerate(raw) for c in conns
            ]

        if self.control_mode == "agent":
            for tls_id in self._controlled_tls_ids:
                if tls_id in self.tls_managers:
                    self.tls_managers[tls_id].apply_to_traci(self.traci)

    def get_observation(self) -> Dict[str, IntersectionObservation]:
        pending_cnt = len(self.traci.simulation.getPendingVehicles())
        obs = {}
        for tl_id, mgr in self.tls_managers.items():
            links = self.tls_controlled_links[tl_id]
            in_lanes = {link.from_lane for link in links}
            out_lanes = {link.to_lane for link in links}
            in_q = {lane: float(self.traci.lane.getLastStepHaltingNumber(lane)) for lane in in_lanes}
            out_q = {lane: float(self.traci.lane.getLastStepHaltingNumber(lane)) for lane in out_lanes}
            in_v = {lane: float(self.traci.lane.getLastStepVehicleNumber(lane)) for lane in in_lanes}
            out_v = {lane: float(self.traci.lane.getLastStepVehicleNumber(lane)) for lane in out_lanes}

            obs[tl_id] = IntersectionObservation(
                tls_id=tl_id,
                current_phase=mgr.get_current_phase(),
                time_in_phase=mgr.time_in_current_phase(),
                is_in_transition=mgr.is_in_transition(),
                green_phases=list(mgr.green_phases),
                controlled_links=links,
                in_lane_queues=in_q,
                out_lane_queues=out_q,
                in_lane_vehicles=in_v,
                out_lane_vehicles=out_v,
                sim_time=self.sim_time,
                step=self.current_step,
                pending_vehicles=pending_cnt,
                valid_actions=mgr.valid_actions(),
            )
        return obs

    def step(self, actions: Optional[Dict[str, int]] = None) -> StepMetrics:
        if not self.is_running:
            raise RuntimeError("SimulationRunner is not running. Call start() first.")

        if actions is None and self.controller is not None:
            decision_steps = max(1, int(round(self.decision_interval / self.config.step_length)))
            if self._last_selected_actions is None or self._steps_since_last_decision >= decision_steps:
                obs = self.get_observation()
                actions = self.controller.select_action({tls: obs[tls] for tls in self._controlled_tls_ids})
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
                raise ValueError("Cannot pass external actions when control_mode='original'.")
            if actions.keys() - self._controlled_tls_ids:
                raise ValueError("Actions must target configured controlled TLS IDs")
            for tls_id, target_phase in actions.items():
                self.tls_managers[tls_id].request_phase(target_phase)
        # Max green fallback enforcement
        for tls_id in self._controlled_tls_ids:
            mgr = self.tls_managers.get(tls_id)
            if mgr and mgr.is_max_green_exceeded() and len(mgr.green_phases) > 1:
                forced_p = (mgr.get_current_phase() + 1) % len(mgr.green_phases)
                mgr.request_phase(forced_p)

        # Apply signals to TraCI
        if self.control_mode == "agent":
            for tls_id in self._controlled_tls_ids:
                if tls_id in self.tls_managers:
                    self.tls_managers[tls_id].apply_to_traci(self.traci)

        # Advance simulation
        self.traci.simulationStep()
        self.current_step += 1
        self.sim_time = self.traci.simulation.getTime()

        # Required safety observations fail the run if they cannot be measured.
        self.total_teleports += int(self.traci.simulation.getStartingTeleportNumber())
        collisions = self.traci.simulation.getCollisions()
        if collisions:
            self.total_collision_steps += 1
            self.total_collision_incidents += len(collisions)
            for collision in collisions:
                involved = [vid for vid in (collision.collider, collision.victim) if vid]
                self.total_collision_vehicle_involvements += len(involved)
                self.unique_colliding_vehicles.update(involved)
        for tls_id, manager in self.tls_managers.items():
            if tls_id in self._controlled_tls_ids:
                manager.update(step_duration=self.config.step_length)
            else:
                manager.sync_external_phase(
                    signal_state=self.traci.trafficlight.getRedYellowGreenState(tls_id),
                    step_duration=self.config.step_length,
                )

        return self.metrics_collector.record_step(
            step=self.current_step,
            sim_time=self.sim_time,
            controlled_lanes=self.all_controlled_lanes,
            traci_module=self.traci,
            step_length=self.config.step_length,
        )

    def safety_metadata(self) -> Dict[str, Any]:
        return {"max_green": self.config.max_green, "safety_events": []}

    def run(
        self,
        num_steps: Optional[int] = None,
        controller: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if controller is not None:
            self.set_controller(controller)

        max_steps = num_steps if num_steps is not None else self.config.max_steps

        try:
            self.start()
            while True:
                if max_steps is not None and self.current_step >= max_steps:
                    break
                if self.sim_time >= self.config.end_time:
                    break
                self.step()
        except BaseException:
            self.close()
            raise
        return self.finalize()

    def finalize(self) -> Dict[str, Any]:
        """Capture final accounting once and always release owned resources."""
        if self._final_summary is None:
            try:
                self._final_summary = self._build_final_summary()
            finally:
                self.close()
        return copy.deepcopy(self._final_summary)

    def _build_final_summary(self) -> Dict[str, Any]:
        """Aggregate accounting before closing SUMO and reading its final XML."""
        running_vehs = len(self.traci.vehicle.getIDList()) if self.is_running else 0
        pending_vehs = len(self.traci.simulation.getPendingVehicles()) if self.is_running else 0

        accounting_meta = build_vehicle_accounting(
            scenario_info=self.scenario_info,
            window_begin=self.config.begin_time,
            window_end=self.sim_time,
            total_completed=self.metrics_collector._cumulative_completed,
            running_vehs=running_vehs,
            pending_vehs=pending_vehs,
            total_teleports=self.total_teleports,
            total_collisions=self.total_collision_incidents,
        )

        metadata = {
            "control_mode": self.control_mode,
            "sim_interface": "libsumo" if self.is_libsumo else "traci",
            "seed": self.config.seed,
            "step_length": self.config.step_length,
            "decision_interval": self.config.decision_interval,
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
            "controlled_tls_ids": sorted(list(self._controlled_tls_ids)),
        }
        metadata.update(accounting_meta)
        if self.controller is not None:
            metadata["controller_name"] = self.controller.name
            metadata["controller_config"] = self.controller.get_config()

        self.close(clean_temp=False)
        metadata.update(self.safety_metadata())
        self.metrics_collector.set_run_metadata(metadata)

        self.metrics_collector.reconcile_tripinfo()

        run_folder = None
        json_path = None
        csv_path = None

        if self.config.save_run_details and self.config.output_dir:
            run_folder = os.path.join(self.config.output_dir, self.run_folder_name)
            os.makedirs(run_folder, exist_ok=True)
            if self.tripinfo_path and os.path.isfile(self.tripinfo_path):
                final_tripinfo = os.path.join(run_folder, "tripinfo.xml")
                shutil.copy2(self.tripinfo_path, final_tripinfo)
                self.metrics_collector.tripinfo_file = final_tripinfo
            json_path, csv_path = self.metrics_collector.export_results(run_folder)

        for path in (self.tripinfo_path, self.sumo_log_path):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

        summary = self.metrics_collector.get_summary()
        summary["output_json"] = json_path
        summary["output_csv"] = csv_path
        summary["output_dir"] = run_folder
        return summary

    def close(self, clean_temp: bool = True) -> None:
        """Close once; cleanup failures cannot replace an active simulation error."""
        active_error = sys.exception()
        try:
            if self.is_running:
                self.is_running = False
                self.traci.close()
        except Exception as error:
            if active_error is None:
                raise
            active_error.add_note(f"SUMO close also failed: {error}")
        finally:
            self.has_run = True
            if clean_temp:
                for path in (self.tripinfo_path, self.sumo_log_path):
                    if path:
                        try:
                            Path(path).unlink(missing_ok=True)
                        except OSError as error:
                            warnings.warn(f"Could not remove temporary output {path}: {error}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
