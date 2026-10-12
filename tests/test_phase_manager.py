import unittest
from src.phase_manager import SafePhaseManager, build_yellow_state


class TestPhaseManager(unittest.TestCase):
    def setUp(self):
        # Example 2-phase green states (e.g. North-South vs East-West)
        self.green_states = [
            "GGGrrrGGGrrr",  # Phase 0
            "rrrGGGrrrGGG",  # Phase 1
        ]
        self.manager = SafePhaseManager(
            tls_id="tls_test",
            green_phases=self.green_states,
            yellow_duration=3.0,
            min_green=5.0,
        )

    def test_initial_state(self):
        self.assertEqual(self.manager.get_current_phase(), 0)
        self.assertEqual(self.manager.get_current_state(), "GGGrrrGGGrrr")
        self.assertFalse(self.manager.is_in_transition())
        self.assertEqual(self.manager.time_in_current_phase(), 0.0)

    def test_build_yellow_state(self):
        s1 = "GGGrrrGGGrrr"
        s2 = "rrrGGGrrrGGG"
        yellow = build_yellow_state(s1, s2)
        # G's that become r's should turn to y
        self.assertEqual(yellow, "yyyrrryyyrrr")

        # Test case where one movement stays green
        s3 = "GGGG"
        s4 = "GGrr"
        self.assertEqual(build_yellow_state(s3, s4), "GGyy")

    def test_min_green_constraint(self):
        # At t=0, current_green_time=0 < min_green (5.0), cannot change
        accepted = self.manager.request_phase(1)
        self.assertFalse(accepted)
        self.assertEqual(self.manager.get_current_phase(), 0)
        self.assertFalse(self.manager.is_in_transition())

        # Advance by 4 seconds (total 4.0s < 5.0s)
        self.manager.update(4.0)
        self.assertFalse(self.manager.request_phase(1))

        # Advance another 1.0 second (total 5.0s == min_green)
        self.manager.update(1.0)
        # Now request should be accepted!
        accepted = self.manager.request_phase(1)
        self.assertTrue(accepted)
        self.assertTrue(self.manager.is_in_transition())
        self.assertEqual(self.manager.get_current_state(), "yyyrrryyyrrr")

    def test_yellow_duration_transition(self):
        # Advance 5 seconds to satisfy min green
        self.manager.update(5.0)
        self.assertTrue(self.manager.request_phase(1))
        self.assertTrue(self.manager.is_in_transition())

        # Step 2.0 seconds in yellow (duration is 3.0s, so 1.0s remaining)
        self.manager.update(2.0)
        self.assertTrue(self.manager.is_in_transition())
        # While in yellow, new phase requests must be rejected
        self.assertFalse(self.manager.request_phase(0))

        # Step remaining 1.0 second
        self.manager.update(1.0)
        # Transition complete! Now in Phase 1
        self.assertFalse(self.manager.is_in_transition())
        self.assertEqual(self.manager.get_current_phase(), 1)
        self.assertEqual(self.manager.get_current_state(), "rrrGGGrrrGGG")
        self.assertEqual(self.manager.time_in_current_phase(), 0.0)

    def test_request_same_phase(self):
        self.manager.update(10.0)
        # Requesting phase 0 when already in phase 0 should return True and not trigger yellow
        accepted = self.manager.request_phase(0)
        self.assertTrue(accepted)
        self.assertFalse(self.manager.is_in_transition())
        self.assertEqual(self.manager.get_current_state(), "GGGrrrGGGrrr")

    def test_invalid_phase_index(self):
        self.manager.update(10.0)
        with self.assertRaises(IndexError):
            self.manager.request_phase(99)


if __name__ == "__main__":
    unittest.main()
