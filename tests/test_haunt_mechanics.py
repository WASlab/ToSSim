import unittest

from Simulation.game import Game
from Simulation.config import GameConfiguration
from Simulation.player import Player
from Simulation.roles import Jester, Doctor, Bodyguard, Sheriff
from Simulation.enums import Attack


class JesterHauntMechanicTests(unittest.TestCase):
    """Ensure Jester haunt cannot be blocked by heal or bodyguard."""

    def setUp(self):
        from Simulation.player import Player as _P
        _P._id_counter = 0
        self.config = GameConfiguration(game_mode="All Any", coven=False)

    def _run_single_night(self, game):
        game.advance_to_night()
        game.process_night_submissions()
        game.advance_to_day()

    def _prepare_lynched_jester(self, jester: Player, target: Player):
        """Mark jester as lynched and assign haunt target before night."""
        jester.is_alive = False
        jester.was_lynched = True
        # Candidates could be players who voted guilty; we include at least the target.
        jester.haunt_candidates = [target]
        jester.haunt_target = target

    def test_doctor_cannot_save_from_haunt(self):
        victim = Player("Victim", Sheriff())
        doc = Player("Doc", Doctor())
        jest = Player("Jest", Jester())

        # Mark jester lynched with haunt set
        self._prepare_lynched_jester(jest, victim)

        game = Game(self.config, [victim, doc, jest])

        # Doctor attempts to heal victim
        game.submit_night_action(doc, victim)

        self._run_single_night(game)

        self.assertFalse(victim.is_alive, "Victim should die to unstoppable haunt despite Doctor heal.")
        self.assertTrue(doc.is_alive, "Doctor should survive since haunt only targets victim.")

    def test_bodyguard_cannot_save_from_haunt(self):
        victim = Player("Victim", Sheriff())
        bg = Player("BG", Bodyguard())
        jest = Player("Jest", Jester())

        self._prepare_lynched_jester(jest, victim)

        game = Game(self.config, [victim, bg, jest])

        # Bodyguard protects victim
        game.submit_night_action(bg, victim)

        self._run_single_night(game)

        self.assertFalse(victim.is_alive, "Victim should still die to unstoppable haunt even with Bodyguard.")
        self.assertTrue(bg.is_alive, "Bodyguard should not die because it cannot intercept Jester haunt.")


if __name__ == "__main__":
    unittest.main() 