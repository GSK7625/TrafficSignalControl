"""Experience Replay Buffer for Deep Q-Network (DQN)."""

from collections import deque
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity experience replay buffer storing environment transitions.

    Stores tuples of:
    - state: np.ndarray of shape (obs_dim,)
    - action: int
    - reward: float
    - next_state: np.ndarray of shape (obs_dim,)
    - done: bool
    - valid_next_mask: np.ndarray of shape (action_dim,) boolean mask
    """

    def __init__(self, capacity: int = 20000, seed: Optional[int] = None):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        if seed is not None:
            random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        valid_next_mask: Optional[np.ndarray] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        """Appends a single transition to the replay buffer."""
        term = bool(terminated) if terminated is not None else bool(done)
        trunc = bool(truncated) if truncated is not None else False
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
                np.array(valid_next_mask, dtype=bool) if valid_next_mask is not None else None,
                term,
                trunc,
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        return_terminated: bool = False,
    ):
        """Samples a random batch of transitions converted to PyTorch tensors.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, next_masks)
            or (states, actions, rewards, next_states, dones, next_masks, terminated_flags).
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}")

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks, terms, truncs = zip(*batch)

        t_states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        t_actions = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        t_rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        t_next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
        t_dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        t_terms = torch.as_tensor(terms, dtype=torch.float32, device=device).unsqueeze(1)

        t_masks = None
        if masks[0] is not None:
            t_masks = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)

        if return_terminated:
            return t_states, t_actions, t_rewards, t_next_states, t_dones, t_masks, t_terms
        return t_states, t_actions, t_rewards, t_next_states, t_dones, t_masks

    def __len__(self) -> int:
        return len(self.buffer)
