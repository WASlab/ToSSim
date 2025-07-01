import unittest

from Simulation.game import Game
from Simulation.config import GameConfiguration
from Simulation.player import Player
from Simulation.roles import Veteran, Mafioso, Werewolf


class VeteranMechanicTests(unittest.TestCase):
    def setUp(self):
        from Simulation.player import Player as _P
        _P._id_counter = 0
        self.config = GameConfiguration(game_mode="All Any", coven=False)

    def _run_single_night(self, game):
        game.advance_to_night()
        game.process_night_submissions()
        game.advance_to_day()

    def test_alert_kills_visitor(self):
        vet = Player("Vet", Veteran())
        maf = Player("Maf", Mafioso())

        game = Game(self.config, [vet, maf])

        # Vet alerts by targeting self
        game.submit_night_action(vet, vet)
        # Mafioso attacks Vet
        game.submit_night_action(maf, vet)

        self._run_single_night(game)

        self.assertTrue(vet.is_alive, "Veteran should survive their own alert night.")
        self.assertFalse(maf.is_alive, "Visitor should be killed by Powerful alert attack.")

    def test_werewolf_rampage_trades_with_veteran(self):
        """On a full-moon night a rampaging Werewolf and an alert Veteran should kill each other."""

        vet = Player("Vet", Veteran())
        wolf = Player("Wolf", Werewolf())

        game = Game(self.config, [vet, wolf])

        # Veteran alerts
        game.submit_night_action(vet, vet)
        # Werewolf chooses Veteran's house for rampage
        game.submit_night_action(wolf, vet)

        # Night 1 in this engine corresponds to game.day == 0 which satisfies the full-moon check.
        self._run_single_night(game)

        self.assertFalse(vet.is_alive, "Veteran should die to Powerful Werewolf maul while on alert.")
        self.assertFalse(wolf.is_alive, "Werewolf should die to Veteran's Powerful alert shot.")


if __name__ == "__main__":
    unittest.main() 