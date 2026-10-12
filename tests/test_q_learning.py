"""Unit tests for Tabular Q-Learning controller."""

import os
import tempfile
import numpy as np
import pytest

from src.controllers.q_learning import QLearningController
from src.rl.discretization import StateDiscretizer


class TestQLearning:
    """Unit tests verifying mathematical correctness and persistence of QLearningController."""

    def test_q_table_initialization_on_demand(self):
        agent = QLearningController(num_actions=4, learning_rate=0.1, gamma=0.9)
        state_1 = (0, 1, 2, 1)

        # Initially table is empty
        assert len(agent.q_table) == 0

        # Querying allocates zeros
        q_vals = agent.get_q_values(state_1)
        assert isinstance(q_vals, np.ndarray)
        assert q_vals.shape == (4,)
        np.testing.assert_array_equal(q_vals, np.zeros(4))
        assert len(agent.q_table) == 1

    def test_bellman_update_non_terminal(self):
        agent = QLearningController(num_actions=3, learning_rate=0.5, gamma=0.8, epsilon=0.0)
        s0 = (0, 0, 0, 1)
        s1 = (1, 0, 1, 1)

        # Setup next state Q-values
        agent.get_q_values(s1)[:] = [2.0, 5.0, 1.0]  # max Q(s1, a') = 5.0

        # Update Q(s0, action=1) with reward = 2.0
        # Target = r + gamma * max_a' Q(s1, a') = 2.0 + 0.8 * 5.0 = 6.0
        # Initial Q(s0, 1) = 0.0
        # New Q(s0, 1) = 0.0 + 0.5 * (6.0 - 0.0) = 3.0
        td_err = agent.update(state=s0, action=1, reward=2.0, next_state=s1, done=False)

        assert td_err == pytest.approx(6.0)
        assert agent.get_q_values(s0)[1] == pytest.approx(3.0)
        assert agent.get_q_values(s0)[0] == pytest.approx(0.0)
        assert agent.get_q_values(s0)[2] == pytest.approx(0.0)

    def test_bellman_update_terminal_no_bootstrap(self):
        agent = QLearningController(num_actions=3, learning_rate=0.2, gamma=0.95, epsilon=0.0)
        s0 = (0, 1, 0, 1)
        s1 = (1, 1, 1, 1)
        agent.get_q_values(s1)[:] = [10.0, 20.0, 30.0]  # even if s1 has high Q

        # When done=True, no bootstrap: Target = reward = -5.0
        # New Q(s0, 0) = 0.0 + 0.2 * (-5.0 - 0.0) = -1.0
        agent.update(state=s0, action=0, reward=-5.0, next_state=s1, done=True)
        assert agent.get_q_values(s0)[0] == pytest.approx(-1.0)

    def test_update_respects_valid_next_actions(self):
        agent = QLearningController(num_actions=3, learning_rate=1.0, gamma=1.0, epsilon=0.0)
        s0 = (0, 0, 0, 1)
        s1 = (1, 0, 0, 1)
        # In s1, action 2 has Q=100.0, but only action 0 (Q=10.0) is valid
        agent.get_q_values(s1)[:] = [10.0, 5.0, 100.0]

        agent.update(
            state=s0,
            action=0,
            reward=0.0,
            next_state=s1,
            done=False,
            valid_next_actions=[0],
        )
        # Should use max among valid: Q=10.0, NOT 100.0
        assert agent.get_q_values(s0)[0] == pytest.approx(10.0)

    def test_action_selection_respects_valid_actions(self):
        agent = QLearningController(num_actions=4, epsilon=0.0)  # purely greedy
        s0 = (0, 0, 0, 1)
        # Highest Q is at action 3 (10.0), but valid actions are only [0, 1]
        agent.get_q_values(s0)[:] = [2.0, 5.0, 1.0, 10.0]

        chosen = agent.select_action_for_state(s0, valid_actions=[0, 1])
        assert chosen == 1  # max within [0, 1]

    def test_epsilon_decay(self):
        agent = QLearningController(num_actions=2, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
        assert agent.decay_epsilon() == pytest.approx(0.5)
        assert agent.decay_epsilon() == pytest.approx(0.25)
        assert agent.decay_epsilon() == pytest.approx(0.125)
        assert agent.decay_epsilon() == pytest.approx(0.1)  # clamped to epsilon_min
        assert agent.decay_epsilon() == pytest.approx(0.1)

    def test_evaluation_mode_freezes_learning(self):
        agent = QLearningController(num_actions=2, learning_rate=0.5, eval_mode=True)
        s0 = (0, 0, 0, 0)
        agent.update(s0, action=0, reward=10.0, next_state=s0, done=True)
        # No updates in eval_mode
        assert agent.get_q_values(s0)[0] == pytest.approx(0.0)
        assert agent.total_updates == 0

    def test_save_and_load_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "q_model.json")
            agent1 = QLearningController(
                num_actions=3,
                learning_rate=0.25,
                gamma=0.9,
                epsilon=0.4,
                epsilon_min=0.05,
                epsilon_decay=0.95,
                seed=123,
            )
            s0 = (0, 1, 2, 1)
            s1 = (1, 0, 1, 2)
            agent1.update(s0, 1, reward=5.0, next_state=s1, done=False)
            agent1.save(model_path)

            # Load into fresh agent
            agent2 = QLearningController(num_actions=3)
            agent2.load(model_path)

            assert agent2.learning_rate == pytest.approx(0.25)
            assert agent2.gamma == pytest.approx(0.9)
            assert agent2.epsilon == pytest.approx(0.4)
            assert agent2.total_updates == 1
            np.testing.assert_array_almost_equal(
                agent2.get_q_values(s0), agent1.get_q_values(s0)
            )
