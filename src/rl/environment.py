"""Reinforcement Learning Environment for Traffic Signal Control in SUMO."""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import SimulationConfig, get_scenario_config
from src.rl.discretization import StateDiscretizer
from src.rl.rewards import BaseRewardFunction, NegativePressureReward
from src.metrics import build_vehicle_accounting
from src.runner import SimulationRunner


class TrafficSignalRLEnv:
    """Gymnasium-compatible Traffic Signal Control Environment."""

    def __init__(
        self,
        scenario_name: str,
        decision_interval: float = 5.0,
        min_green: float = 5.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 0.0,
        max_green: Optional[float] = 60.0,
        max_steps: Optional[int] = None,
        seed: int = 42,
        reward_fn: Optional[BaseRewardFunction] = None,
        output_dir: str = "results/rl",
        use_tripinfo: bool = True,
        gui: bool = False,
        gui_delay: float = 0.0,
        state_discretizer: Optional[StateDiscretizer] = None,
        max_effective_green: float = 25.0,
        serving_queue_thresholds: Optional[Tuple[float, float]] = None,
        waiting_queue_thresholds: Optional[Tuple[float, float, float]] = None,
        target_tls_id: Optional[str] = None,
        save_run_details: bool = False,
        **kwargs: Any,
    ):
        self.scenario_name = scenario_name
        self.decision_interval = float(decision_interval)
        self.min_green = float(min_green)
        self.max_green = float(max_green) if max_green is not None else None
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.max_steps = max_steps
        self.seed = int(seed)
        self.output_dir = output_dir
        self.use_tripinfo = use_tripinfo
        self.gui = gui
        self.gui_delay = float(gui_delay)
        self.save_run_details = save_run_details

        self.config: SimulationConfig = get_scenario_config(
            dataset_name=scenario_name,
            gui=gui,
            gui_delay=self.gui_delay,
            step_length=1.0,
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
            min_green=min_green,
            max_green=max_green,
            max_steps=max_steps,
            seed=seed,
            output_dir=output_dir,
            control_mode="agent",
            use_tripinfo=use_tripinfo,
            decision_interval=self.decision_interval,
            save_run_details=save_run_details,
        )

        self.runner = SimulationRunner(self.config)
        self.tls_id = target_tls_id or self.runner.scenario_info.tls_ids[0]
        self.target_tls_id = self.tls_id
        self.config.target_tls_id = self.tls_id
        self.config.controlled_tls_ids = [self.tls_id]
        self.runner.set_controlled_tls_ids(self.tls_id)

        self.green_phases = self.runner.tls_managers[self.tls_id].green_phases
        self.num_actions = len(self.green_phases)
        self.incoming_lanes = self.runner.scenario_info.controlled_lanes_per_tls.get(
            self.tls_id, self.runner.all_controlled_lanes
        )

        self.state_discretizer = state_discretizer or StateDiscretizer(
            min_green=self.min_green,
            max_effective_green=max_effective_green,
            serving_queue_thresholds=serving_queue_thresholds or (1.0, 5.0),
            waiting_queue_thresholds=waiting_queue_thresholds or (1.0, 6.0, 15.0),
        )
        self.reward_fn = reward_fn or NegativePressureReward(scale=1.0)

        self.observation_dim = 4
        self.action_dim = self.num_actions
        self._episode_count = 0
        self._last_summary: Optional[Dict[str, Any]] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Starts or resets a simulation episode."""
        if seed is not None:
            self.seed = seed
            self.config.seed = seed

        self.close()
        self.runner.reset()
        self.runner.set_controlled_tls_ids(self.tls_id)
        self.runner.start()

        self._episode_count += 1
        self.reward_fn.reset()

        tls_obs = self.runner.get_observation()[self.tls_id]
        discrete_state = self.state_discretizer.discretize(tls_obs)

        info = {
            "tls_id": self.tls_id,
            "discrete_state": discrete_state,
            "valid_actions": self.get_valid_actions(),
            "action_mask": self.get_valid_action_mask(),
            "raw_observation": tls_obs,
            "sim_time": self.runner.sim_time,
            "step": self.runner.current_step,
            "episode": self._episode_count,
        }
        return discrete_state, info

    def get_valid_actions(self) -> List[int]:
        """Returns valid action indices based on signal timing constraints."""
        if not getattr(self.runner, "is_running", False):
            return list(range(self.num_actions))

        mgr = self.runner.tls_managers[self.tls_id]
        curr = mgr.get_current_phase()

        if mgr.is_in_transition():
            return [mgr.get_target_phase()]
        if mgr.time_in_current_phase() < self.min_green:
            return [curr]
        if self.max_green and mgr.time_in_current_phase() >= self.max_green and self.num_actions > 1:
            return [a for a in range(self.num_actions) if a != curr]

        return list(range(self.num_actions))

    def get_valid_action_mask(self) -> np.ndarray:
        """Returns boolean mask where True denotes a valid action."""
        mask = np.zeros(self.num_actions, dtype=bool)
        for a in self.get_valid_actions():
            if 0 <= a < self.num_actions:
                mask[a] = True
        return mask

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Applies control action and advances simulation by decision_interval seconds."""
        action = int(action)
        self.reward_fn.reset()

        steps_to_run = max(1, int(round(self.decision_interval / self.config.step_length)))
        terminated = False
        truncated = False

        for step_idx in range(steps_to_run):
            if self.runner.sim_time >= self.config.end_time:
                terminated = True
                break
            if self.config.max_steps and self.runner.current_step >= self.config.max_steps:
                truncated = True
                break

            act_dict = {self.tls_id: action} if step_idx == 0 and 0 <= action < self.num_actions else {}
            step_metrics = self.runner.step(actions=act_dict)

            tls_obs = self.runner.get_observation()[self.tls_id]
            pending = getattr(step_metrics, "pending_vehicles", 0) if step_metrics else 0
            self.reward_fn.record_step(
                in_lane_queues=tls_obs.in_lane_queues,
                out_lane_queues=tls_obs.out_lane_queues,
                pending_vehicles=pending,
                controlled_links=tls_obs.controlled_links,
            )

        reward = self.reward_fn.calculate_reward()
        tls_obs = self.runner.get_observation()[self.tls_id]
        discrete_state = self.state_discretizer.discretize(tls_obs)

        if self.runner.sim_time >= self.config.end_time:
            terminated = True
        if self.config.max_steps and self.runner.current_step >= self.config.max_steps:
            truncated = True

        info = {
            "tls_id": self.tls_id,
            "discrete_state": discrete_state,
            "valid_actions": self.get_valid_actions(),
            "action_mask": self.get_valid_action_mask(),
            "raw_observation": tls_obs,
            "sim_time": self.runner.sim_time,
            "step": self.runner.current_step,
        }

        if terminated or truncated:
            self._last_summary = self._finalize_episode()
            info["episode_summary"] = self._last_summary

        return discrete_state, reward, terminated, truncated, info

    def _finalize_episode(self) -> Dict[str, Any]:
        """Finalizes episode, collects metrics, and exports results."""
        running_vehs = 0
        pending_vehs = 0
        try:
            if getattr(self.runner, "traci", None) and self.runner.is_running:
                running_vehs = len(self.runner.traci.vehicle.getIDList())
                pending_vehs = len(self.runner.traci.simulation.getPendingVehicles())
        except Exception:
            pass

        accounting_meta = build_vehicle_accounting(
            scenario_name=self.config.dataset_name,
            window_begin=self.runner.config.begin_time,
            window_end=self.runner.sim_time,
            step_length=self.config.step_length,
            total_completed=self.runner.metrics_collector._cumulative_completed,
            running_vehs=running_vehs,
            pending_vehs=pending_vehs,
            total_teleports=self.runner.total_teleports,
            total_collisions=self.runner.total_collision_incidents,
            scenario_info=self.runner.scenario_info,
            population=self.runner.metrics_collector.population_snapshot(),
        )

        metadata = {
            "control_mode": "agent",
            "sim_interface": "libsumo" if getattr(self.runner, "is_libsumo", False) else "traci",
            "seed": self.config.seed,
            "step_length": self.config.step_length,
            "yellow_duration": self.config.yellow_duration,
            "all_red_duration": self.config.all_red_duration,
            "min_green": self.config.min_green,
            "decision_interval": self.decision_interval,
            "collision_count": self.runner.total_collision_incidents,
            "phase_change_count": sum(m.phase_switch_count for m in self.runner.tls_managers.values()),
        }
        metadata.update(accounting_meta)
        metadata.update(self.runner.safety_metadata())
        self.runner.close()
        self.runner.metrics_collector.set_run_metadata(metadata)

        # Reconcile tripinfo from tempfile if present
        if self.runner.tripinfo_path and os.path.isfile(self.runner.tripinfo_path):
            self.runner.metrics_collector.tripinfo_file = self.runner.tripinfo_path
            self.runner.metrics_collector.reconcile_tripinfo()

        run_folder = None
        json_path = None
        csv_path = None

        if getattr(self.config, "save_run_details", False) and getattr(self.config, "output_dir", None):
            run_folder = os.path.join(self.config.output_dir, self.runner.run_folder_name)
            os.makedirs(run_folder, exist_ok=True)
            if self.runner.tripinfo_path and os.path.isfile(self.runner.tripinfo_path):
                final_tripinfo = os.path.join(run_folder, "tripinfo.xml")
                shutil.copy2(self.runner.tripinfo_path, final_tripinfo)
                self.runner.metrics_collector.tripinfo_file = final_tripinfo
            json_path, csv_path = self.runner.metrics_collector.export_results(run_folder)

        if self.runner.tripinfo_path and os.path.isfile(self.runner.tripinfo_path):
            try:
                os.remove(self.runner.tripinfo_path)
            except Exception:
                pass

        summary = self.runner.metrics_collector.get_summary()
        summary["output_json"] = json_path
        summary["output_csv"] = csv_path
        summary["output_dir"] = run_folder
        self._last_summary = summary
        return summary

    def get_summary(self) -> Optional[Dict[str, Any]]:
        return self._last_summary

    def close(self) -> None:
        if self.runner is not None and getattr(self.runner, "is_running", False):
            self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


TrafficSignalEnv = TrafficSignalRLEnv
