"""Unit tests for MaxPressureController."""

import unittest
from src.controllers.base import ControlledLink, IntersectionObservation
from src.controllers.max_pressure import MaxPressureController, compute_phase_pressure


class TestMaxPressureController(unittest.TestCase):
    def setUp(self):
        # Setup standard 2-phase intersection with 4 links
        # Phase 0 ('GGrr'): links 0 & 1 green (in_0 -> out_0, in_1 -> out_1)
        # Phase 1 ('rrGG'): links 2 & 3 green (in_2 -> out_2, in_3 -> out_3)
        self.links = [
            ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_0"),
            ControlledLink(signal_index=1, from_lane="in_1", to_lane="out_1"),
            ControlledLink(signal_index=2, from_lane="in_2", to_lane="out_2"),
            ControlledLink(signal_index=3, from_lane="in_3", to_lane="out_3"),
        ]
        self.green_phases = ["GGrr", "rrGG"]

    def test_basic_pressure_calculation(self):
        """Case 1: Basic formula P(i,j) = q_in(i) - q_out(j)."""
        in_queues = {"in_0": 10.0, "in_1": 5.0, "in_2": 2.0, "in_3": 1.0}
        out_queues = {"out_0": 2.0, "out_1": 1.0, "out_2": 0.0, "out_3": 0.0}

        # Phase 0: (10 - 2) + (5 - 1) = 8 + 4 = 12.0
        p0 = compute_phase_pressure(
            phase_state=self.green_phases[0],
            controlled_links=self.links,
            in_lane_metrics=in_queues,
            out_lane_metrics=out_queues,
        )
        self.assertEqual(p0, 12.0)

        # Phase 1: (2 - 0) + (1 - 0) = 3.0
        p1 = compute_phase_pressure(
            phase_state=self.green_phases[1],
            controlled_links=self.links,
            in_lane_metrics=in_queues,
            out_lane_metrics=out_queues,
        )
        self.assertEqual(p1, 3.0)

    def test_in_vs_out_distinction(self):
        """Case 2: Distinguish incoming from outgoing lanes."""
        # If in_0 has 10 and out_0 has 2 -> pressure is +8
        p_normal = compute_phase_pressure(
            phase_state="Grr",
            controlled_links=[ControlledLink(0, "in_0", "out_0")],
            in_lane_metrics={"in_0": 10.0},
            out_lane_metrics={"out_0": 2.0},
        )
        self.assertEqual(p_normal, 8.0)

        # If swapped: in_0 has 2 and out_0 has 10 -> pressure is -8
        p_swapped = compute_phase_pressure(
            phase_state="Grr",
            controlled_links=[ControlledLink(0, "in_0", "out_0")],
            in_lane_metrics={"in_0": 2.0},
            out_lane_metrics={"out_0": 10.0},
        )
        self.assertEqual(p_swapped, -8.0)

    def test_zero_queue(self):
        """Case 3: All queues zero yields zero pressure."""
        p_zero = compute_phase_pressure(
            phase_state=self.green_phases[0],
            controlled_links=self.links,
            in_lane_metrics={"in_0": 0.0, "in_1": 0.0},
            out_lane_metrics={"out_0": 0.0, "out_1": 0.0},
        )
        self.assertEqual(p_zero, 0.0)

    def test_negative_pressure(self):
        """Case 4: Severe downstream congestion produces negative pressure."""
        p_neg = compute_phase_pressure(
            phase_state=self.green_phases[0],
            controlled_links=self.links,
            in_lane_metrics={"in_0": 2.0, "in_1": 1.0},
            out_lane_metrics={"out_0": 15.0, "out_1": 10.0},
        )
        # (2 - 15) + (1 - 10) = -13 + -9 = -22.0
        self.assertEqual(p_neg, -22.0)

    def test_multiple_connections_sharing_same_incoming_lane(self):
        """Case 5: Multiple outgoing connections from a single incoming lane do not double-count q_in."""
        # in_0 connects to both out_A (straight) and out_B (left)
        # Both are green in Phase 0
        multi_links = [
            ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_A"),
            ControlledLink(signal_index=1, from_lane="in_0", to_lane="out_B"),
        ]
        # in_0 has 10 vehicles waiting, out_A has 1, out_B has 2
        # Without protection, summing naively would give (10 - 1) + (10 - 2) = 17 (in_0 counted twice = 20)
        # With proper weighting (1/2 each or unique lane aggregation):
        # Inflow: 10 * (1/2 + 1/2) = 10; Outflow: 1 + 2 = 3. Total pressure = 10 - 3 = 7.0
        p = compute_phase_pressure(
            phase_state="GG",
            controlled_links=multi_links,
            in_lane_metrics={"in_0": 10.0},
            out_lane_metrics={"out_A": 1.0, "out_B": 2.0},
        )
        self.assertEqual(p, 7.0)

    def test_multiple_connections_merging_into_same_outgoing_lane(self):
        """Case 6: Multiple incoming lanes merging into same outgoing lane do not double-count backpressure."""
        # in_0 and in_1 both connect to out_common
        merging_links = [
            ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_common"),
            ControlledLink(signal_index=1, from_lane="in_1", to_lane="out_common"),
        ]
        # in_0 has 6, in_1 has 8, out_common has 4
        # Total inflow = 6 + 8 = 14; Total backpressure = 4. Net pressure = 14 - 4 = 10.0
        p = compute_phase_pressure(
            phase_state="GG",
            controlled_links=merging_links,
            in_lane_metrics={"in_0": 6.0, "in_1": 8.0},
            out_lane_metrics={"out_common": 4.0},
        )
        self.assertEqual(p, 10.0)

    def test_boundary_links_with_no_tracked_outgoing_lane(self):
        """Case 7: Boundary links exit network without tracked outgoing congestion (q_out=0)."""
        boundary_links = [
            ControlledLink(signal_index=0, from_lane="in_0", to_lane=""),
        ]
        p = compute_phase_pressure(
            phase_state="G",
            controlled_links=boundary_links,
            in_lane_metrics={"in_0": 5.0},
            out_lane_metrics={},
        )
        self.assertEqual(p, 5.0)

    def test_tie_breaking_favors_current_phase(self):
        """Case 8: When multiple phases have identical maximum pressure, keep current phase."""
        ctrl = MaxPressureController(min_green=5.0)
        ctrl.reset()

        obs = IntersectionObservation(
            tls_id="tls_test",
            current_phase=1,
            time_in_phase=10.0,
            is_in_transition=False,
            green_phases=self.green_phases,
            controlled_links=self.links,
            in_lane_queues={"in_0": 0.0, "in_1": 0.0, "in_2": 0.0, "in_3": 0.0},
            out_lane_queues={"out_0": 0.0, "out_1": 0.0, "out_2": 0.0, "out_3": 0.0},
            in_lane_vehicles={},
            out_lane_vehicles={},
            sim_time=100.0,
            step=10,
        )

        # Both phase 0 and phase 1 have pressure 0.0. Current phase is 1.
        # Tie-breaker must retain phase 1 to avoid needless yellow transition!
        action = ctrl.select_action({"tls_test": obs})
        self.assertEqual(action["tls_test"], 1)

    def test_min_green_constraint_prevents_premature_switch(self):
        """Case 9: Even if another phase has higher pressure, do not switch before min_green."""
        ctrl = MaxPressureController(min_green=10.0)
        ctrl.reset()

        # Phase 0 has huge queue (pressure ~ 50), phase 1 has 0
        # Current phase is 1, but time_in_phase is only 4.0 (< 10.0 min_green)
        obs = IntersectionObservation(
            tls_id="tls_test",
            current_phase=1,
            time_in_phase=4.0,
            is_in_transition=False,
            green_phases=self.green_phases,
            controlled_links=self.links,
            in_lane_queues={"in_0": 50.0, "in_1": 0.0, "in_2": 0.0, "in_3": 0.0},
            out_lane_queues={"out_0": 0.0, "out_1": 0.0, "out_2": 0.0, "out_3": 0.0},
            in_lane_vehicles={},
            out_lane_vehicles={},
            sim_time=20.0,
            step=20,
        )

        action = ctrl.select_action({"tls_test": obs})
        self.assertEqual(action["tls_test"], 1)  # Must hold phase 1 because min_green not met

        # Once time_in_phase >= 10.0, switch to phase 0 with higher pressure
        obs.time_in_phase = 10.0
        action = ctrl.select_action({"tls_test": obs})
        self.assertEqual(action["tls_test"], 0)

    def test_three_phase_selection(self):
        """Case 10: Select highest pressure phase among 3 candidate phases."""
        links = [
            ControlledLink(0, "lane_N", "lane_S"),
            ControlledLink(1, "lane_E", "lane_W"),
            ControlledLink(2, "lane_Left", "lane_Exit"),
        ]
        green_phases = ["Grr", "rGr", "rrG"]
        ctrl = MaxPressureController(min_green=5.0)
        ctrl.reset()

        obs = IntersectionObservation(
            tls_id="tls_3p",
            current_phase=0,
            time_in_phase=6.0,
            is_in_transition=False,
            green_phases=green_phases,
            controlled_links=links,
            in_lane_queues={"lane_N": 2.0, "lane_E": 1.0, "lane_Left": 15.0},
            out_lane_queues={"lane_S": 0.0, "lane_W": 0.0, "lane_Exit": 0.0},
            in_lane_vehicles={},
            out_lane_vehicles={},
            sim_time=50.0,
            step=50,
        )

        action = ctrl.select_action({"tls_3p": obs})
        self.assertEqual(action["tls_3p"], 2)  # Phase 2 has highest pressure (15.0)


if __name__ == "__main__":
    unittest.main()
