"""Unit tests for StarvationWatchdog and anti-starvation mechanisms."""

from src.controllers.base import ControlledLink
from src.phase_manager import SafePhaseManager, StarvationWatchdog


class TestStarvationWatchdog:
    """Test suite for StarvationWatchdog."""

    def test_watchdog_initialization(self):
        watchdog = StarvationWatchdog(green_phases=["GGGrrr", "rrrGGG"], max_red=90.0)
        assert watchdog.max_red == 90.0
        assert watchdog.has_starvation() is False

    def test_watchdog_filters_actions_when_branch_is_starved(self):
        # 2 phases: Phase 0 serves edgeA, Phase 1 serves edgeB
        phases = ["GGGggrrrrr", "rrrrrGGGgg"]
        links = [
            ControlledLink(0, "edgeA_0", "out_0"),
            ControlledLink(5, "edgeB_0", "out_1"),
        ]
        watchdog = StarvationWatchdog(green_phases=phases, controlled_links=links, max_red=60.0)
        
        # Initially edgeB has queue, but red time is 0
        watchdog.update_queues({"edgeB_0": 3.0})
        watchdog.update(step_duration=30.0, current_state=phases[0], is_in_transition=False)
        assert watchdog.has_starvation() is False
        assert watchdog.filter_valid_actions([0, 1]) == [0, 1]
        
        # Advance red timer past 60s
        watchdog.update(step_duration=35.0, current_state=phases[0], is_in_transition=False)
        assert watchdog.has_starvation() is True
        # Must force action to phase 1 (the one that serves edgeB)
        assert watchdog.filter_valid_actions([0, 1]) == [1]

    def test_watchdog_ignores_starved_phase_if_no_queue(self):
        phases = ["GGGggrrrrr", "rrrrrGGGgg"]
        links = [
            ControlledLink(0, "edgeA_0", "out_0"),
            ControlledLink(5, "edgeB_0", "out_1"),
        ]
        watchdog = StarvationWatchdog(green_phases=phases, controlled_links=links, max_red=60.0)
        # edgeB has 0 queue
        watchdog.update_queues({"edgeB_0": 0.0})
        watchdog.update(step_duration=70.0, current_state=phases[0], is_in_transition=False)
        
        # Since queue is 0, no starvation
        assert watchdog.has_starvation() is False
        assert watchdog.filter_valid_actions([0, 1]) == [0, 1]

    def test_safe_phase_manager_with_max_red(self):
        phases = ["Grr", "rGr", "rrG"]
        links = [
            ControlledLink(0, "edgeA_0", "out_0"),
            ControlledLink(1, "edgeB_0", "out_1"),
            ControlledLink(2, "edgeC_0", "out_2"),
        ]
        mgr = SafePhaseManager("tls", phases, min_green=5.0, max_green=60.0, max_red=90.0)
        mgr.configure_links(links)
        
        # Initially in phase 0, min_green = 5s
        assert mgr.valid_actions() == [0]
        mgr.update(5.0)
        assert mgr.valid_actions() == [0, 1, 2]
        
        # Edge C has cars waiting
        mgr.update_queues({"edgeC_0": 2.0})
        mgr.update(90.0)
        # Phase 2 (serving Edge C) must now be enforced
        assert mgr.valid_actions() == [2]

    def test_watchdog_uses_exact_branch_names_and_current_queues(self):
        watchdog = StarvationWatchdog(["Gr", "rG"], [
            ControlledLink(0, "edge_0", "out_0"), ControlledLink(1, "edge2_0", "out_1"),
        ], max_red=10)
        watchdog.update(20, "rr", True)
        watchdog.update_queues({"edge2_0": 2})
        assert [branch for _, branch in watchdog.get_starving_branches()] == ["edge2"]
        watchdog.update_queues({})
        # An observed empty snapshot clears the previous queue.
        assert watchdog.in_lane_queues == {}
        assert not watchdog.has_starvation()

    def test_watchdog_timer_is_not_reset_before_green_is_applied(self):
        mgr = SafePhaseManager("tls", ["Gr", "rG"], min_green=0, yellow_duration=3, max_red=5)
        mgr.configure_links([ControlledLink(1, "edge_0", "out_0")])
        mgr.update(5)
        assert mgr.request_phase(1)
        mgr.update(3)
        assert mgr.get_current_state() == "rG"
        assert mgr.watchdog.branch_red_timers["edge"] == 8
        mgr.update(1)
        assert mgr.watchdog.branch_red_timers["edge"] == 0

