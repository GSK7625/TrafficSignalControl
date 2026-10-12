"""Deep Q-Network (DQN) Traffic Signal Controller using PyTorch."""

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.controllers.base import BaseTrafficController, IntersectionObservation
from src.rl.observations import ObservationBuilder
from src.rl.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Multi-Layer Perceptron (MLP) Q-value function approximator."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNController(BaseTrafficController):
    """Deep Q-Network (DQN) controller for traffic signal control.

    Key Components:
    - Online Q-Network and Target Q-Network.
    - Experience Replay Buffer.
    - Epsilon-greedy exploration with decay schedule.
    - Valid action masking during exploration, greedy action selection, and Bellman targets.
    - Periodic target network weight synchronization.
    - Gradient norm clipping for numerical stability.
    - Checkpoint saving and loading.
    - Evaluation mode (freezes weights and disables exploration).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.90,
        buffer_size: int = 20000,
        batch_size: int = 32,
        learning_starts: int = 64,
        target_update_interval: int = 100,
        max_grad_norm: float = 10.0,
        obs_builder: Optional[ObservationBuilder] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        eval_mode: bool = False,
    ):
        super().__init__(name="DQN")
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)
        self.obs_builder = obs_builder
        self.eval_mode = bool(eval_mode)

        self.device = torch.device(device if device is not None else "cpu")

        if seed is not None:
            self.set_seed(seed)

        # Networks
        self.q_network = QNetwork(self.observation_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network = QNetwork(self.observation_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability against outlier queues

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size, seed=seed)

        # Metrics tracking
        self.total_steps: int = 0
        self.total_updates: int = 0
        self.action_counts: np.ndarray = np.zeros(self.action_dim, dtype=int)

    def set_seed(self, seed: int) -> None:
        """Sets random seed for PyTorch, NumPy, and random module."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        """Resets execution history."""
        self.history.clear()

    def select_action_for_state(
        self,
        obs_vec: np.ndarray,
        valid_actions: Optional[Sequence[int]] = None,
    ) -> int:
        """Selects an action given observation vector using epsilon-greedy exploration.

        Args:
            obs_vec: Normalized observation vector of shape (observation_dim,).
            valid_actions: Optional list of valid actions.

        Returns:
            Chosen action index.
        """
        if valid_actions is None or len(valid_actions) == 0:
            candidates = list(range(self.action_dim))
        else:
            candidates = [a for a in valid_actions if 0 <= a < self.action_dim]
            if not candidates:
                candidates = list(range(self.action_dim))

        if self.eval_mode or random.random() >= self.epsilon:
            # Exploitation: argmax Q over valid candidates
            with torch.no_grad():
                t_obs = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_network(t_obs).squeeze(0).cpu().numpy()

                candidate_q = [q_values[a] for a in candidates]
                max_q = max(candidate_q)
                best_candidates = [
                    a for a, q in zip(candidates, candidate_q) if abs(q - max_q) <= 1e-6
                ]
                chosen_action = random.choice(best_candidates)
        else:
            # Exploration: uniform choice over valid candidates
            chosen_action = random.choice(candidates)

        self.action_counts[chosen_action] += 1
        return chosen_action

    def train_step(self) -> Optional[float]:
        """Samples a batch from the replay buffer and performs a single gradient update."""
        if self.eval_mode:
            return None

        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(
            self.batch_size, device=self.device
        )

        # 1. Current Q-values: Q(s, a)
        curr_q = self.q_network(states).gather(1, actions)

        # 2. Target Q-values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            if next_masks is not None:
                # Mask out invalid next actions with very low value
                next_q = next_q.masked_fill(~next_masks, -1e9)
            max_next_q = next_q.max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        # 3. Compute loss and optimize
        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.total_updates += 1

        # 4. Periodic target network update
        if self.total_updates % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> float:
        """Decays exploration probability down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """BaseTrafficController interface method: maps observations to target actions."""
        actions: Dict[str, int] = {}
        for tls_id, obs in observations.items():
            if self.obs_builder is None:
                # Fallback to current phase if builder not configured
                actions[tls_id] = obs.current_phase
                continue

            obs_vec = self.obs_builder.build_vector(obs)

            if obs.is_in_transition:
                valid_actions = [obs.current_phase]
            elif obs.time_in_phase < self.obs_builder.min_green:
                valid_actions = [obs.current_phase]
            else:
                valid_actions = list(range(len(obs.green_phases)))

            act = self.select_action_for_state(obs_vec, valid_actions=valid_actions)
            actions[tls_id] = act

            self.log_decision(
                tls_id=tls_id,
                sim_time=obs.sim_time,
                step=obs.step,
                action=act,
                details={
                    "epsilon": self.epsilon,
                    "eval_mode": self.eval_mode,
                },
            )
        return actions

    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        return {
            "name": self.name,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "target_update_interval": self.target_update_interval,
            "eval_mode": self.eval_mode,
            "obs_builder_config": self.obs_builder.get_config() if self.obs_builder else None,
        }

    def save(self, filepath: str) -> None:
        """Saves PyTorch weights, optimizer state, and hyperparameters to checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.get_config(),
            "total_updates": self.total_updates,
            "total_steps": self.total_steps,
            "action_counts": self.action_counts.tolist(),
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """Loads model weights, optimizer, and hyperparameters from checkpoint."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])

        if "optimizer_state_dict" in checkpoint and not self.eval_mode:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        cfg = checkpoint["config"]
        self.learning_rate = cfg.get("learning_rate", self.learning_rate)
        self.gamma = cfg.get("gamma", self.gamma)
        self.epsilon = cfg.get("epsilon", self.epsilon)
        self.epsilon_min = cfg.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = cfg.get("epsilon_decay", self.epsilon_decay)
        self.total_updates = checkpoint.get("total_updates", 0)

        if "action_counts" in checkpoint:
            self.action_counts = np.array(checkpoint["action_counts"], dtype=int)

        if self.eval_mode:
            self.q_network.eval()
            self.target_network.eval()
