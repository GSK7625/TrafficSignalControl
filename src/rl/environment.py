"""Reinforcement Learning Environment for Traffic Signal Control in SUMO."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import DEFAULT_SAFETY_CONFIG, SimulationConfig, get_scenario_config
from src.rl.discretization import StateDiscretizer
from src.rl.rewards import BaseRewardFunction, NegativePressureReward
from src.runner import SimulationRunner


class TrafficSignalRLEnv:
    """Gymnasium-compatible Traffic Signal Control Environment."""

    def __init__(
        self,
        scenario_name: str,
        decision_interval: float = 5.0,
        min_green: float = DEFAULT_SAFETY_CONFIG["min_green"],
        yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"],
        all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"],
        max_green: Optional[float] = DEFAULT_SAFETY_CONFIG["max_green"],
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
        step_length: float = 1.0,
    ):
        self.decision_interval = float(decision_interval)
        self.min_green = float(min_green)
        self.max_green = float(max_green) if max_green is not None else None
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.seed = int(seed)
        self.gui_delay = float(gui_delay)

        self.config: SimulationConfig = get_scenario_config(
            dataset_name=scenario_name,
            gui=gui,
            gui_delay=self.gui_delay,
            step_length=step_length,
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
        self.runner.set_controlled_tls_ids(self.tls_id)

        self.green_phases = self.runner.tls_managers[self.tls_id].green_phases
        self.num_actions = len(self.green_phases)
        self.state_discretizer = state_discretizer or StateDiscretizer(
            min_green=self.min_green,
            max_effective_green=max_effective_green,
            serving_queue_thresholds=serving_queue_thresholds or (1.0, 5.0),
            waiting_queue_thresholds=waiting_queue_thresholds or (1.0, 6.0, 15.0),
        )
        self.reward_fn = reward_fn or NegativePressureReward(scale=1.0)
        self.reward_config_source = "explicit" if reward_fn else "defaults"

        self._episode_count = 0
        self._episode_done = True
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

        self._last_summary = None
        self._episode_done = False
        self._episode_count += 1
        self.reward_fn.reset()

        info = self._observation_info()
        info["episode"] = self._episode_count
        return info["discrete_state"], info

    def _observation_info(self) -> Dict[str, Any]:
        """Build the shared observation fields for reset and step."""
        tls_obs = self.runner.get_observation()[self.tls_id]
        return {
            "tls_id": self.tls_id,
            "discrete_state": self.state_discretizer.discretize(tls_obs),
            "valid_actions": self.get_valid_actions(),
            "action_mask": self.get_valid_action_mask(),
            "raw_observation": tls_obs,
            "sim_time": self.runner.sim_time,
            "step": self.runner.current_step,
        }

    def get_valid_actions(self) -> List[int]:
        """Returns valid action indices based on signal timing constraints."""
        if not self.runner.is_running:
            return list(range(self.num_actions))

        return self.runner.tls_managers[self.tls_id].valid_actions()

    def get_valid_action_mask(self) -> np.ndarray:
        """Returns boolean mask where True denotes a valid action."""
        mask = np.zeros(self.num_actions, dtype=bool)
        for a in self.get_valid_actions():
            if 0 <= a < self.num_actions:
                mask[a] = True
        return mask

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Applies control action and advances simulation by decision_interval seconds."""
        if self._episode_done or not self.runner.is_running:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.num_actions:
            raise ValueError(f"Action must be an integer in [0, {self.num_actions})")
        self.reward_fn.reset()

        steps_to_run = max(1, int(round(self.decision_interval / self.config.step_length)))
        terminated = False
        truncated = False

        for step_idx in range(steps_to_run):
            if self.runner.sim_time >= self.config.end_time:
                terminated = True
                break
            if self.config.max_steps is not None and self.runner.current_step >= self.config.max_steps:
                truncated = True
                break

            act_dict = {self.tls_id: action} if step_idx == 0 else {}
            step_metrics = self.runner.step(actions=act_dict)

            tls_obs = self.runner.get_observation()[self.tls_id]
            pending = step_metrics.pending_vehicles
            self.reward_fn.record_step(
                in_lane_queues=tls_obs.in_lane_queues,
                out_lane_queues=tls_obs.out_lane_queues,
                pending_vehicles=pending,
                controlled_links=tls_obs.controlled_links,
            )

        reward = self.reward_fn.calculate_reward()
        info = self._observation_info()

        if self.runner.sim_time >= self.config.end_time:
            terminated = True
        if self.config.max_steps is not None and self.runner.current_step >= self.config.max_steps:
            truncated = True

        if terminated or truncated:
            self._episode_done = True
            self.runner.metrics_collector.set_run_metadata({
                "reward_config": self.reward_fn.get_config(),
                "reward_config_source": self.reward_config_source,
            })
            self._last_summary = self.runner.finalize()
            info["episode_summary"] = self._last_summary

        return info["discrete_state"], reward, terminated, truncated, info

    def get_summary(self) -> Optional[Dict[str, Any]]:
        return self._last_summary

    def close(self) -> None:
        self.runner.close()
        self._episode_done = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


TrafficSignalEnv = TrafficSignalRLEnv
