"""Unit tests for Deep Q-Network (DQN) controller."""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.controllers.dqn import DQNController, QNetwork
from src.rl.replay_buffer import ReplayBuffer


class TestDQN:
    """Unit tests for DQN network, replay buffer, updates, and persistence."""

    def test_q_network_forward_pass_and_shape(self):
        input_dim = 13
        output_dim = 3
        net = QNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=64)

        x = torch.randn(5, input_dim)
        out = net(x)
        assert out.shape == (5, output_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_replay_buffer_push_and_sample(self):
        buf = ReplayBuffer(capacity=100, seed=42)
        assert len(buf) == 0

        state = np.ones(10, dtype=np.float32)
        next_state = np.zeros(10, dtype=np.float32)
        mask = np.array([True, False, True])

        for i in range(20):
            buf.push(state, action=i % 3, reward=float(i), next_state=next_state, done=(i == 19), valid_next_mask=mask)

        assert len(buf) == 20
        b_s, b_a, b_r, b_ns, b_d, b_m = buf.sample(batch_size=8)

        assert b_s.shape == (8, 10)
        assert b_a.shape == (8, 1)
        assert b_r.shape == (8, 1)
        assert b_ns.shape == (8, 10)
        assert b_d.shape == (8, 1)
        assert b_m.shape == (8, 3)

    def test_dqn_action_selection_respects_valid_actions(self):
        agent = DQNController(
            observation_dim=4, action_dim=3, epsilon_start=0.0, eval_mode=True
        )
        obs = np.array([0.5, 0.2, 0.1, 0.9], dtype=np.float32)

        # Force weights such that action 2 has highest Q
        with torch.no_grad():
            agent.q_network.network[-1].bias.copy_(torch.tensor([1.0, 2.0, 10.0]))

        # When only [0, 1] are valid, action 2 must NOT be chosen
        chosen = agent.select_action_for_state(obs, valid_actions=[0, 1])
        assert chosen == 1  # max among [0, 1]

    def test_dqn_training_step_reduces_loss(self):
        agent = DQNController(
            observation_dim=4,
            action_dim=2,
            learning_rate=1e-2,
            batch_size=8,
            learning_starts=8,
            target_update_interval=10,
            seed=42,
        )

        obs = np.ones(4, dtype=np.float32)
        next_obs = np.zeros(4, dtype=np.float32)
        mask = np.array([True, True])

        # Push 16 identical transitions with consistent target
        for _ in range(16):
            agent.replay_buffer.push(obs, action=0, reward=1.0, next_state=next_obs, done=True, valid_next_mask=mask)

        loss1 = agent.train_step()
        assert loss1 is not None
        assert isinstance(loss1, float)
        assert loss1 >= 0.0

        for _ in range(20):
            agent.train_step()

        loss_final = agent.train_step()
        assert loss_final < loss1  # Loss should decrease on consistent target

    def test_target_network_periodic_update(self):
        agent = DQNController(
            observation_dim=2,
            action_dim=2,
            batch_size=4,
            learning_starts=4,
            target_update_interval=5,
            seed=42,
        )

        for _ in range(10):
            agent.replay_buffer.push(
                np.ones(2, dtype=np.float32), 0, 1.0, np.ones(2, dtype=np.float32), False, np.array([True, True])
            )

        # Before updates, weights are identical
        p_online_init = next(agent.q_network.parameters()).clone()
        p_target_init = next(agent.target_network.parameters()).clone()
        torch.testing.assert_close(p_online_init, p_target_init)

        # Run 2 train steps (total_updates=2 < 5)
        agent.train_step()
        agent.train_step()
        p_online_after = next(agent.q_network.parameters()).clone()
        p_target_after = next(agent.target_network.parameters()).clone()

        # Target should still be at init
        torch.testing.assert_close(p_target_after, p_target_init)
        assert not torch.equal(p_online_after, p_online_init)

        # Run 3 more steps (total_updates reaches 5)
        agent.train_step()
        agent.train_step()
        agent.train_step()
        assert agent.total_updates == 5

        # Now target network should be updated to online network weights
        p_online_sync = next(agent.q_network.parameters()).clone()
        p_target_sync = next(agent.target_network.parameters()).clone()
        torch.testing.assert_close(p_online_sync, p_target_sync)

    def test_dqn_save_and_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "dqn_test.pt")
            agent1 = DQNController(
                observation_dim=5,
                action_dim=3,
                learning_rate=5e-4,
                gamma=0.98,
                epsilon_start=0.7,
                seed=999,
            )

            # Change online weights
            with torch.no_grad():
                for p in agent1.q_network.parameters():
                    p.add_(1.0)

            agent1.save(model_path)

            agent2 = DQNController(
                observation_dim=5,
                action_dim=3,
                seed=111,
                eval_mode=True,
            )
            agent2.load(model_path)

            assert agent2.learning_rate == pytest.approx(5e-4)
            assert agent2.gamma == pytest.approx(0.98)
            assert agent2.epsilon == pytest.approx(0.7)

            # Test output identical
            test_x = torch.randn(2, 5)
            out1 = agent1.q_network(test_x)
            out2 = agent2.q_network(test_x)
            torch.testing.assert_close(out1, out2)
