"""Reinforcement Learning Environment for Traffic Signal Control in SUMO."""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import SimulationConfig, get_scenario_config
from src.controllers.base import IntersectionObservation
from src.rl.discretization import StateDiscretizer
from src.rl.observations import ObservationBuilder
from src.rl.rewards import BaseRewardFunction, NegativeAverageQueueReward
from src.runner import SimulationRunner


class TrafficSignalRLEnv:
    """Gymnasium-compatible Traffic Signal Control Environment.

    Shared by Tabular Q-Learning and Deep Q-Network (DQN).

    Key Characteristics:
    - Wraps SimulationRunner, SafePhaseManager, and MetricsCollector without duplicating logic.
    - Operates at fixed decision intervals (default: 5.0 seconds).
    - Accurately tracks requested action vs executed phase vs clearance transitions.
    - Provides valid action masking to strictly respect minimum green times and clearance safety.
    - Handles SUMO lifecycle cleanly across multiple episodes.
    """

    def __init__(
        self,
        scenario_name: str,
        decision_interval: float = 5.0,
        min_green: float = 5.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 0.0,
        max_steps: Optional[int] = None,
        seed: int = 42,
        reward_fn: Optional[BaseRewardFunction] = None,
        output_dir: str = "results/rl",
        use_tripinfo: bool = True,
        gui: bool = False,
        obs_builder: Optional[ObservationBuilder] = None,
        state_discretizer: Optional[StateDiscretizer] = None,
        max_queue_scale: float = 20.0,
        max_time_scale: float = 60.0,
        max_effective_green: float = 25.0,
        serving_queue_thresholds: Optional[Tuple[float, float]] = None,
        waiting_queue_thresholds: Optional[Tuple[float, float, float]] = None,
        target_tls_id: Optional[str] = None,
    ):
        self.scenario_name = scenario_name
        self.target_tls_id = target_tls_id
        d_int = float(decision_interval)
        if d_int <= 0 or d_int != d_int or d_int == float("inf"):
            raise ValueError(f"decision_interval must be positive and finite, got {decision_interval}")
        sim_step_length = 1.0
        if d_int < sim_step_length:
            raise ValueError(
                f"decision_interval ({decision_interval}s) must be >= simulation step_length ({sim_step_length}s)"
            )
        ratio = d_int / sim_step_length
        if abs(ratio - round(ratio)) > 1e-5:
            raise ValueError(
                f"decision_interval ({decision_interval}s) must be an integer multiple of simulation step_length ({sim_step_length}s)"
            )
        self.decision_interval = d_int
        self.min_green = float(min_green)
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.max_steps = max_steps
        self.seed = int(seed)
        self.output_dir = output_dir
        self.use_tripinfo = use_tripinfo
        self.gui = gui

        # Initialize SimulationConfig and temporary runner to inspect intersection details
        self.config: SimulationConfig = get_scenario_config(
            dataset_name=scenario_name,
            gui=gui,
            step_length=1.0,
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
            min_green=min_green,
            max_steps=max_steps,
            seed=seed,
            output_dir=output_dir,
            control_mode="agent",
            use_tripinfo=use_tripinfo,
            decision_interval=self.decision_interval,
        )

        self.runner = SimulationRunner(self.config)
        all_tls = self.runner.scenario_info.tls_ids
        if len(all_tls) > 1 and target_tls_id is None:
            raise ValueError(
                f"Scenario '{scenario_name}' contains {len(all_tls)} traffic lights ({all_tls}). "
                f"TrafficSignalRLEnv is a single-agent environment and requires single-intersection networks "
                f"or an explicit target_tls_id. Multi-agent RL is required for coordinated multi-intersection control."
            )
        self.tls_id = target_tls_id if target_tls_id is not None else all_tls[0]
        if self.tls_id not in self.runner.tls_managers:
            raise ValueError(f"Target TLS '{self.tls_id}' not found in scenario '{scenario_name}'")
        # Restrict runner's active control strictly to the controlled intersection
        self.runner._controlled_tls_ids = {self.tls_id}
        self.green_phases = self.runner.tls_managers[self.tls_id].green_phases
        self.num_actions = len(self.green_phases)

        # Incoming lanes controlled by this intersection
        self.incoming_lanes = self.runner.scenario_info.controlled_lanes_per_tls.get(
            self.tls_id, []
        )
        if not self.incoming_lanes:
            self.incoming_lanes = self.runner.all_controlled_lanes

        # Build feature extractors
        if obs_builder is not None:
            self.obs_builder = obs_builder
        else:
            self.obs_builder = ObservationBuilder(
                incoming_lanes=self.incoming_lanes,
                num_green_phases=self.num_actions,
                min_green=self.min_green,
                max_queue_scale=max_queue_scale,
                max_time_scale=max_time_scale,
            )

        if state_discretizer is not None:
            self.state_discretizer = state_discretizer
        else:
            disc_kwargs: Dict[str, Any] = {
                "min_green": self.min_green,
                "max_effective_green": max_effective_green,
            }
            if serving_queue_thresholds is not None:
                disc_kwargs["serving_queue_thresholds"] = serving_queue_thresholds
            if waiting_queue_thresholds is not None:
                disc_kwargs["waiting_queue_thresholds"] = waiting_queue_thresholds
            self.state_discretizer = StateDiscretizer(**disc_kwargs)

        # Reward function
        self.reward_fn: BaseRewardFunction = (
            reward_fn if reward_fn is not None else NegativeAverageQueueReward(scale=1.0)
        )

        self.observation_dim: int = self.obs_builder.observation_dim
        self.action_dim: int = self.num_actions

        self._current_step_count: int = 0
        self._episode_count: int = 0
        self._last_obs_dict: Dict[str, IntersectionObservation] = {}
        self._last_summary: Optional[Dict[str, Any]] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Starts or resets a simulation episode.

        Returns:
            Tuple of (continuous_observation_vector, info_dict).
        """
        if seed is not None:
            self.seed = seed
            self.config.seed = seed

        # Ensure previous simulation is cleanly closed
        self.close()

        # Reset runner and initialize fresh run
        self.runner.reset()
        self.runner.start()

        self._current_step_count = 0
        self._episode_count += 1
        self.reward_fn.reset()

        raw_obs = self.runner.get_observation()
        self._last_obs_dict = raw_obs
        tls_obs = raw_obs[self.tls_id]

        obs_vec = self.obs_builder.build_vector(tls_obs)
        discrete_state = self.state_discretizer.discretize(tls_obs)

        info = {
            "tls_id": self.tls_id,
            "discrete_state": discrete_state,
            "raw_observation": tls_obs,
            "valid_actions": self.get_valid_actions(),
            "action_mask": self.get_valid_action_mask(),
            "sim_time": self.runner.sim_time,
            "step": self.runner.current_step,
            "episode": self._episode_count,
        }

        return obs_vec, info

    def get_valid_actions(self) -> List[int]:
        """Returns the list of valid action indices based on current signal state and constraints."""
        if not self.runner.is_running:
            return list(range(self.num_actions))

        mgr = self.runner.tls_managers[self.tls_id]
        curr_phase = mgr.get_current_phase()

        # If in clearance transition, agent cannot alter the ongoing transition
        if mgr.is_in_transition():
            return [mgr.get_target_phase()]

        # If minimum green constraint has not yet elapsed, must hold current green phase
        if mgr.time_in_current_phase() < self.min_green:
            return [curr_phase]

        # Otherwise, all valid green phases are eligible for selection
        return list(range(self.num_actions))

    def get_valid_action_mask(self) -> np.ndarray:
        """Returns a boolean array of shape (num_actions,) where True denotes a valid action."""
        mask = np.zeros(self.num_actions, dtype=bool)
        valid = self.get_valid_actions()
        for a in valid:
            if 0 <= a < self.num_actions:
                mask[a] = True
        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies control action and advances simulation by decision_interval seconds.

        Args:
            action: Requested green phase index in [0, num_actions - 1].

        Returns:
            Tuple of (next_obs_vector, reward, terminated, truncated, info).
        """
        if not self.runner.is_running:
            raise RuntimeError("Environment is not running. Call reset() first.")

        action = int(action)
        valid_actions = self.get_valid_actions()
        is_valid = action in valid_actions

        mgr = self.runner.tls_managers[self.tls_id]
        requested_phase = action
        prev_phase = mgr.get_current_phase()

        # Attempt to apply action through SafePhaseManager
        action_accepted = False
        if is_valid:
            action_accepted = mgr.request_phase(action)
        else:
            # Action invalid according to constraints: held safely by manager
            action_accepted = False

        # Reset reward accumulator for this decision interval
        self.reward_fn.reset()

        # Advance simulation for decision_interval steps (e.g. 5 steps of 1.0s)
        steps_to_run = max(1, int(round(self.decision_interval / self.config.step_length)))
        steps_executed = 0
        terminated = False
        truncated = False

        for _ in range(steps_to_run):
            if self.runner.sim_time >= self.config.end_time:
                terminated = True
                break
            if self.config.max_steps is not None and self.runner.current_step >= self.config.max_steps:
                truncated = True
                break

            # Advance 1 step
            step_metrics = self.runner.step(actions=None)
            steps_executed += 1
            self._current_step_count += 1

            # Accumulate queue metrics for reward calculation
            raw_obs = self.runner.get_observation()
            tls_obs = raw_obs[self.tls_id]
            self.reward_fn.record_step(tls_obs.in_lane_queues)

        # Calculate interval reward
        reward = self.reward_fn.calculate_reward()

        # Query updated observation
        raw_obs = self.runner.get_observation()
        self._last_obs_dict = raw_obs
        tls_obs = raw_obs[self.tls_id]

        obs_vec = self.obs_builder.build_vector(tls_obs)
        discrete_state = self.state_discretizer.discretize(tls_obs)

        # Check termination at the end of the interval
        if self.runner.sim_time >= self.config.end_time:
            terminated = True
        if self.config.max_steps is not None and self.runner.current_step >= self.config.max_steps:
            truncated = True

        info = {
            "tls_id": self.tls_id,
            "requested_action": requested_phase,
            "action_accepted": action_accepted,
            "is_valid_action": is_valid,
            "prev_phase": prev_phase,
            "actual_phase": mgr.get_current_phase(),
            "is_in_transition": mgr.is_in_transition(),
            "time_in_phase": mgr.time_in_current_phase(),
            "steps_in_interval": steps_executed,
            "discrete_state": discrete_state,
            "raw_observation": tls_obs,
            "valid_actions": self.get_valid_actions(),
            "action_mask": self.get_valid_action_mask(),
            "sim_time": self.runner.sim_time,
            "step": self.runner.current_step,
        }

        if terminated or truncated:
            # Finalize run and export summary
            self._last_summary = self._finalize_episode()
            info["episode_summary"] = self._last_summary

        return obs_vec, reward, terminated, truncated, info

    def _finalize_episode(self) -> Dict[str, Any]:
        """Finalizes the simulation run and retrieves the summary from MetricsCollector."""
        running_vehs = 0
        pending_vehs = 0
        try:
            import traci
            if traci is not None and self.runner.is_running:
                running_vehs = len(traci.vehicle.getIDList())
                pending_vehs = len(traci.simulation.getPendingVehicles())
        except Exception:
            pass

        metadata = {
            "control_mode": "agent",
            "seed": self.config.seed,
            "step_length": self.config.step_length,
            "yellow_duration": self.config.yellow_duration,
            "all_red_duration": self.config.all_red_duration,
            "min_green": self.config.min_green,
            "decision_interval": self.decision_interval,
            "running_vehicles_at_end": running_vehs,
            "pending_insertion_vehicles": pending_vehs,
            "teleport_count": self.runner.total_teleports,
            "collision_count": self.runner.total_collision_incidents,
            "collision_incidents": self.runner.total_collision_incidents,
            "collision_steps": self.runner.total_collision_steps,
            "collision_vehicle_involvements": self.runner.total_collision_vehicle_involvements,
            "unique_colliding_vehicles_count": len(self.runner.unique_colliding_vehicles),
            "phase_change_count": sum(mgr.phase_switch_count for mgr in self.runner.tls_managers.values()),
            "phase_durations_per_tls": {tl_id: mgr.phase_durations for tl_id, mgr in self.runner.tls_managers.items()},
        }
        self.runner.metrics_collector.set_run_metadata(metadata)
        self.runner.close()

        # Export metrics into unique run folder
        run_folder = os.path.join(self.config.output_dir, self.runner.run_folder_name)
        os.makedirs(run_folder, exist_ok=True)

        if self.runner.tripinfo_path and os.path.isfile(self.runner.tripinfo_path):
            import shutil
            final_tripinfo = os.path.join(run_folder, "tripinfo.xml")
            shutil.copy2(self.runner.tripinfo_path, final_tripinfo)
            try:
                os.remove(self.runner.tripinfo_path)
            except Exception:
                pass
            self.runner.metrics_collector.tripinfo_file = final_tripinfo

        json_path, csv_path = self.runner.metrics_collector.export_results(run_folder)

        summary = self.runner.metrics_collector.get_summary()
        summary["output_json"] = json_path
        summary["output_csv"] = csv_path
        summary["output_dir"] = run_folder
        self._last_summary = summary
        return summary

    def get_summary(self) -> Optional[Dict[str, Any]]:
        """Returns the summary of the last completed episode."""
        return self._last_summary

    def close(self) -> None:
        """Safely closes TraCI and frees simulation resources."""
        if self.runner is not None and self.runner.is_running:
            self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Alias for compatibility
TrafficSignalEnv = TrafficSignalRLEnv
