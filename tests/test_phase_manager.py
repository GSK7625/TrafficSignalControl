"""Unit tests for SafePhaseManager and phase clearance transition logic."""

import pytest
from unittest.mock import MagicMock
from src.phase_manager import SafePhaseManager, build_yellow_state


class TestBuildYellowState:
    """Test suite for build_yellow_state function."""

    def test_length_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="State lengths do not match"):
            build_yellow_state("GGGrrr", "GGrr")

    def test_green_to_red_switches_to_yellow(self):
        # Current: GGG rrr -> Target: rrr GGG
        # First 3: G -> r => y
        # Last 3:  r -> G => r (yellow transition keeps red until green phase activates)
        curr = "GGGrrr"
        target = "rrrGGG"
        yellow = build_yellow_state(curr, target)
        assert yellow == "yyyrrr"

    def test_retained_green_stays_green(self):
        # 1st link stays G, 2nd link goes to red, 3rd link stays red
        curr = "GGr"
        target = "GrG"
        yellow = build_yellow_state(curr, target)
        assert yellow == "Gyr"

    def test_existing_yellow_retained(self):
        curr = "yyyrrr"
        target = "rrrrrr"
        yellow = build_yellow_state(curr, target)
        assert yellow == "yyyrrr"

    def test_s_phase_transitions_safely(self):
        # When current has right-of-way 's' and target is 'r', it must transition through yellow 'y'
        curr = "sGr"
        target = "rrr"
        yellow = build_yellow_state(curr, target)
        assert yellow == "yyr"

        # When current is 'r' and target is 's', it must remain 'r' during yellow clearance
        curr = "rGr"
        target = "srr"
        yellow = build_yellow_state(curr, target)
        assert yellow == "ryr"

        # When current is 's' and target is 'G' or 'g', it holds 's' during yellow transition
        curr = "sGr"
        target = "GGr"
        yellow = build_yellow_state(curr, target)
        assert yellow == "sGr"

        # When current is 'G' and target is 's', it transitions through yellow 'y'
        curr = "GGr"
        target = "sGr"
        yellow = build_yellow_state(curr, target)
        assert yellow == "yGr"

    def test_g_and_G_priority_transitions(self):
        # When transitioning g -> G, holds safer permissive 'g' during yellow
        curr = "gGr"
        target = "GGr"
        yellow = build_yellow_state(curr, target)
        assert yellow == "gGr"

        # When transitioning G -> g, holds safer permissive 'g' during yellow
        curr = "GGr"
        target = "gGr"
        yellow = build_yellow_state(curr, target)
        assert yellow == "gGr"


class TestSafePhaseManager:
    """Test suite for SafePhaseManager class."""

    @pytest.fixture
    def manager(self):
        green_phases = [
            "GGGrrrGGGrrr",  # Phase 0: N-S Green
            "rrrGGGrrrGGG",  # Phase 1: E-W Green
        ]
        return SafePhaseManager(
            tls_id="tls_1",
            green_phases=green_phases,
            yellow_duration=3.0,
            all_red_duration=0.0,
            min_green=5.0,
            initial_phase=0,
        )

    @pytest.fixture
    def manager_with_all_red(self):
        green_phases = [
            "GGGrrr",
            "rrrGGG",
        ]
        return SafePhaseManager(
            tls_id="tls_2",
            green_phases=green_phases,
            yellow_duration=3.0,
            all_red_duration=2.0,
            min_green=5.0,
            initial_phase=0,
        )

    def test_initialization_invalid_arguments(self):
        with pytest.raises(ValueError, match="green_phases cannot be empty"):
            SafePhaseManager("tls", [])

        with pytest.raises(ValueError, match="yellow_duration must be positive"):
            SafePhaseManager("tls", ["GGG"], yellow_duration=0.0)

        with pytest.raises(ValueError, match="all_red_duration must be non-negative"):
            SafePhaseManager("tls", ["GGG"], all_red_duration=-1.0)

        with pytest.raises(ValueError, match="min_green must be non-negative"):
            SafePhaseManager("tls", ["GGG"], min_green=-2.0)

        with pytest.raises(IndexError, match="out of bounds"):
            SafePhaseManager("tls", ["GGG"], initial_phase=5)

    def test_initial_state(self, manager):
        assert manager.get_current_phase() == 0
        assert manager.get_target_phase() == 0
        assert manager.get_current_state() == "GGGrrrGGGrrr"
        assert manager.is_in_transition() is False
        assert manager.time_in_current_phase() == 0.0
        assert manager.phase_switch_count == 0

    def test_request_same_phase_accepted_immediately(self, manager):
        assert manager.request_phase(0) is True
        assert manager.is_in_transition() is False
        assert manager.phase_switch_count == 0

    def test_request_out_of_bounds_phase_raises_error(self, manager):
        with pytest.raises(IndexError, match="out of range"):
            manager.request_phase(5)

    def test_request_phase_rejected_before_min_green(self, manager):
        # At t=0, time_in_phase=0 < min_green=5.0
        assert manager.request_phase(1) is False
        assert manager.get_current_phase() == 0
        assert manager.is_in_transition() is False

        # Advance to t=4.0s (still < 5.0s)
        manager.update(step_duration=4.0)
        assert manager.time_in_current_phase() == 4.0
        assert manager.request_phase(1) is False
        assert manager.is_in_transition() is False

    def test_request_phase_accepted_after_min_green(self, manager):
        # Advance past min_green (5.0s)
        manager.update(step_duration=5.0)
        assert manager.time_in_current_phase() == 5.0

        accepted = manager.request_phase(1)
        assert accepted is True
        assert manager.is_in_transition() is True
        assert manager.get_target_phase() == 1
        assert manager.get_current_state() == "yyyrrryyyrrr"
        assert manager.phase_switch_count == 1
        assert manager.time_in_current_phase() == 0.0

    def test_request_phase_rejected_during_transition(self, manager):
        manager.update(step_duration=5.0)
        manager.request_phase(1)
        assert manager.is_in_transition() is True

        # Trying to request phase 0 or phase 1 again during yellow is rejected
        assert manager.request_phase(0) is False
        assert manager.request_phase(1) is False

    def test_yellow_transition_completion_direct_to_green(self, manager):
        # Reach min green and initiate yellow transition
        manager.update(step_duration=5.0)
        manager.request_phase(1)

        # Yellow duration = 3.0s
        # Step 1: 1s into yellow
        manager.update(step_duration=1.0)
        assert manager.is_in_transition() is True
        assert manager.actual_yellow_time == 1.0
        assert manager.total_transition_time == 1.0

        # Step 2: 2 more seconds into yellow (total 3.0s yellow)
        manager.update(step_duration=2.0)
        assert manager.is_in_transition() is False
        assert manager.get_current_phase() == 1
        assert manager.get_current_state() == "rrrGGGrrrGGG"
        assert manager.actual_yellow_time == 3.0
        assert manager.total_transition_time == 3.0

    def test_yellow_and_all_red_clearance_cycle(self, manager_with_all_red):
        mgr = manager_with_all_red
        # Reach min_green
        mgr.update(step_duration=5.0)
        mgr.request_phase(1)
        assert mgr.is_in_transition() is True
        assert mgr.get_current_state() == "yyyrrr"

        # Advance through 3s yellow
        mgr.update(step_duration=3.0)
        # Should now be in ALL_RED clearance state
        assert mgr.is_in_transition() is True
        assert mgr.get_current_state() == "rrrrrr"

        # Advance 1s in all-red
        mgr.update(step_duration=1.0)
        assert mgr.is_in_transition() is True
        assert mgr.get_current_state() == "rrrrrr"

        # Advance remaining 1s all-red (total 2s all-red)
        mgr.update(step_duration=1.0)
        # Transition complete -> target Green phase 1
        assert mgr.is_in_transition() is False
        assert mgr.get_current_phase() == 1
        assert mgr.get_current_state() == "rrrGGG"
        assert mgr.actual_yellow_time == 3.0
        assert mgr.total_transition_time == 5.0  # 3s yellow + 2s all-red

    def test_sync_external_phase(self, manager):
        # Sync with green state
        manager.sync_external_phase(signal_state="rrrGGGrrrGGG", step_duration=1.0)
        assert manager.get_current_phase() == 1
        assert manager.is_in_transition() is False
        assert manager.phase_durations[1] == 1.0

        # Sync with yellow state
        manager.sync_external_phase(signal_state="yyyrrryyyrrr", step_duration=1.0)
        assert manager.is_in_transition() is True
        assert manager.actual_yellow_time == 1.0
        assert manager.total_transition_time == 1.0

    def test_apply_to_traci(self, manager):
        mock_traci = MagicMock()
        manager.apply_to_traci(mock_traci)
        mock_traci.trafficlight.setRedYellowGreenState.assert_called_once_with(
            "tls_1", "GGGrrrGGGrrr"
        )

    def test_apply_to_traci_raises_runtime_error_on_failure(self, manager):
        mock_traci = MagicMock()
        mock_traci.trafficlight.setRedYellowGreenState.side_effect = Exception("Connection closed")
        with pytest.raises(RuntimeError, match="Failed to set state for TLS 'tls_1'"):
            manager.apply_to_traci(mock_traci)

    def test_max_green_enforces_phase_switch(self):
        green_phases = ["GGGrrr", "rrrGGG"]
        mgr = SafePhaseManager(
            tls_id="tls_1",
            green_phases=green_phases,
            yellow_duration=3.0,
            min_green=5.0,
            max_green=60.0,
            initial_phase=0,
        )
        for _ in range(60):
            mgr.update(step_duration=1.0)
        assert mgr.is_max_green_exceeded() is True
        # Requesting to stay in phase 0 when max_green is exceeded must be rejected
        assert mgr.request_phase(0) is False
        # Requesting to switch to phase 1 must be accepted
        assert mgr.request_phase(1) is True

