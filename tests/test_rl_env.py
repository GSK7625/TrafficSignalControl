"""Integration tests for TrafficSignalRLEnv with live SUMO simulation."""

import numpy as np
import pytest

from src.rl.environment import TrafficSignalRLEnv


class TestTrafficSignalRLEnv:
    """Live integration tests for the unified RL Environment."""

    def test_env_lifecycle_cologne1(self):
        env = TrafficSignalRLEnv(
            scenario_name="cologne1",
            decision_interval=5.0,
            min_green=5.0,
            max_steps=20,
            seed=42,
        )

        try:
            assert env.num_actions == 8
            assert env.observation_dim == 8 + 8 + 3  # 8 in-lanes + 8 phases + 3 scalars = 19

            obs, info = env.reset()
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (env.observation_dim,)
            assert not np.isnan(obs).any()
            assert "discrete_state" in info
            assert "valid_actions" in info
            assert len(info["action_mask"]) == env.num_actions

            # Step with valid action 0
            next_obs, reward, terminated, truncated, step_info = env.step(action=0)
            assert isinstance(next_obs, np.ndarray)
            assert next_obs.shape == (env.observation_dim,)
            assert isinstance(reward, float)
            assert reward <= 0.0  # Negative average queue
            assert not np.isnan(next_obs).any()
            assert "steps_in_interval" in step_info
            assert step_info["steps_in_interval"] == 5

        finally:
            env.close()

    def test_env_lifecycle_ingolstadt1(self):
        env = TrafficSignalRLEnv(
            scenario_name="ingolstadt1",
            decision_interval=5.0,
            min_green=5.0,
            max_steps=20,
            seed=42,
        )

        try:
            assert env.num_actions == 3
            assert env.observation_dim == 7 + 3 + 3  # 7 in-lanes + 3 phases + 3 scalars = 13

            obs, info = env.reset()
            assert obs.shape == (env.observation_dim,)

            # Execute 2 decision intervals (10 simulation seconds)
            for action in [0, 1]:
                next_obs, reward, terminated, truncated, step_info = env.step(action=action)
                assert next_obs.shape == (env.observation_dim,)
                assert reward <= 0.0

        finally:
            env.close()

    def test_action_masking_and_min_green_constraint(self):
        env = TrafficSignalRLEnv(
            scenario_name="ingolstadt1",
            decision_interval=2.0,  # 2s interval with min_green=5s will show constraint
            min_green=5.0,
            max_steps=30,
            seed=42,
        )

        try:
            obs, info = env.reset()

            # At start (t=0), min_green is not satisfied: only current phase 0 is valid
            assert env.get_valid_actions() == [0]

            # Step 1 (t=2s): still under min_green (5s) -> only phase 0 valid
            env.step(action=0)
            assert env.get_valid_actions() == [0]

            # Step 2 (t=4s): still under min_green -> only phase 0 valid
            env.step(action=0)
            assert env.get_valid_actions() == [0]

            # Step 3 (t=6s): min_green (5s) now satisfied! All phases valid
            env.step(action=0)
            assert sorted(env.get_valid_actions()) == [0, 1, 2]

            # Now request switch to phase 1 -> initiates yellow transition (3.0s)
            obs1, r1, term1, trunc1, info1 = env.step(action=1)
            assert info1["action_accepted"] is True

            # During transition: only target phase 1 is valid
            if env.runner.tls_managers[env.tls_id].is_in_transition():
                assert env.get_valid_actions() == [1]
                assert env.get_valid_action_mask()[1] == True

        finally:
            env.close()

    def test_invalid_action_rejected_safely(self):
        env = TrafficSignalRLEnv(
            scenario_name="ingolstadt1",
            decision_interval=5.0,
            min_green=5.0,
            max_steps=20,
            seed=42,
        )

        try:
            env.reset()
            # Action 99 is invalid
            obs, reward, term, trunc, info = env.step(action=99)
            assert info["is_valid_action"] is False
            assert info["action_accepted"] is False
            # Simulation should continue safely without crash
            assert not term
        finally:
            env.close()
