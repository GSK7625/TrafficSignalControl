"""Reinforcement Learning Environment for Traffic Signal Control in SUMO."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import DEFAULT_SAFETY_CONFIG, SimulationConfig, get_scenario_config
from src.rl.discretization import StateDiscretizer
from src.rl.rewards import NegativePressureReward
from src.runner import SimulationRunner


class TrafficSignalRLEnv:
    """Decision-interval wrapper for tabular training and evaluation."""

    def __init__(
        self,
        config_or_scenario: Optional[Union[SimulationConfig, str]] = None,
        state_discretizer: Optional[StateDiscretizer] = None,
        reward_fn: Optional[NegativePressureReward] = None,
        target_tls_id: Optional[str] = None,
        # Common scenario configuration parameters for backward compatibility
        scenario_name: Optional[str] = None,
        decision_interval: float = 5.0,
        min_green: float = DEFAULT_SAFETY_CONFIG["min_green"],
        yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"],
        all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"],
        max_green: Optional[float] = DEFAULT_SAFETY_CONFIG["max_green"],
        max_red: Optional[float] = None,
        max_steps: Optional[int] = None,
        seed: int = 42,
        output_dir: str = "results/rl",
        use_tripinfo: bool = True,
        gui: bool = False,
        gui_delay: float = 0.0,
        save_run_details: bool = False,
        step_length: float = 1.0,
        **kwargs: Any,
    ):
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"TrafficSignalRLEnv.__init__() got an unexpected keyword argument '{unexpected}'")

        if isinstance(config_or_scenario, SimulationConfig):
            self.config = config_or_scenario
            self.decision_interval = float(self.config.decision_interval)
            self.min_green = float(self.config.min_green)
            self.max_green = float(self.config.max_green) if self.config.max_green is not None else None
            self.seed = int(self.config.seed)
        else:
            name = config_or_scenario or scenario_name
            if not name:
                raise ValueError("Must provide either a SimulationConfig or a scenario_name")
            self.decision_interval = float(decision_interval)
            self.min_green = float(min_green)
            self.max_green = float(max_green) if max_green is not None else None
            self.seed = int(seed)

            self.config = get_scenario_config(
                dataset_name=str(name),
                gui=gui,
                gui_delay=float(gui_delay),
                step_length=step_length,
                yellow_duration=yellow_duration,
                all_red_duration=all_red_duration,
                min_green=min_green,
                max_green=max_green,
                max_red=max_red,
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
        self.runner.set_controlled_tls_ids(self.tls_id)

        self.green_phases = self.runner.tls_managers[self.tls_id].green_phases
        self.num_actions = len(self.green_phases)
        self.state_discretizer = state_discretizer or StateDiscretizer(min_green=self.min_green)
        self.reward_fn = reward_fn or NegativePressureReward(scale=1.0)
        self.reward_config_source = "explicit" if reward_fn else "defaults"

        self._episode_count = 0
        self._episode_done = True
        self._last_summary: Optional[Dict[str, Any]] = None

    @property
    def target_tls_id(self) -> str:
        return self.tls_id

    @property
    def _is_terminated(self) -> bool:
        return self.runner.sim_time >= self.config.end_time

    @property
    def _is_truncated(self) -> bool:
        return self.config.max_steps is not None and self.runner.current_step >= self.config.max_steps

    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        """Starts or resets a simulation episode."""
        if seed is not None:
            self.seed = seed
            self.config.seed = seed

        self.close()
        self.runner.reset()
        self.runner.set_controlled_tls_ids(self.tls_id)
        self.runner.start()

        links = self.runner.tls_controlled_links.get(self.tls_id)
        if links:
            self.reward_fn.set_topology_from_links(links)

        self._last_summary = None
        self._episode_done = False
        self._episode_count += 1
        self.reward_fn.reset()

        info = self._observation_info()
        info["episode"] = self._episode_count
        return info["discrete_state"], info

    def _observation_info(self, tls_obs=None) -> Dict[str, Any]:
        """Build the shared observation fields for reset and step."""
        if tls_obs is None:
            tls_obs = self.runner.get_observation()[self.tls_id]
        return {
            "tls_id": self.tls_id,
            "discrete_state": self.state_discretizer.discretize(tls_obs),
            "valid_actions": tls_obs.valid_actions,
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
        mask[self.get_valid_actions()] = True
        return mask

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, bool, Dict[str, Any]]:
        """Applies control action and advances simulation by decision_interval seconds."""
        if self._episode_done or not self.runner.is_running:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.num_actions:
            raise ValueError(f"Action must be an integer in [0, {self.num_actions})")

        self.reward_fn.reset_interval()
        steps_to_run = max(1, int(round(self.decision_interval / self.config.step_length)))
        tls_obs = None

        for step_idx in range(steps_to_run):
            if self._is_terminated or self._is_truncated:
                break

            act_dict = {self.tls_id: action} if step_idx == 0 else {}
            step_metrics = self.runner.step(actions=act_dict)

            tls_obs = self.runner.get_observation()[self.tls_id]
            self.reward_fn.record_step(
                in_lane_queues=tls_obs.in_lane_queues,
                out_lane_queues=tls_obs.out_lane_queues,
                pending_vehicles=step_metrics.pending_vehicles,
            )

        terminated, truncated = self._is_terminated, self._is_truncated
        info = self._observation_info(tls_obs)

        if terminated or truncated:
            self._episode_done = True
            self.runner.metrics_collector.set_run_metadata({
                "reward_config": self.reward_fn.get_config(),
                "reward_config_source": self.reward_config_source,
            })
            self._last_summary = self.runner.finalize()
            info["episode_summary"] = self._last_summary

        return info["discrete_state"], self.reward_fn.calculate_reward(), terminated, truncated, info

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
