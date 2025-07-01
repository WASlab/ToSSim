import unittest

from Simulation.game import Game
from Simulation.config import GameConfiguration
from Simulation.player import Player
from Simulation.roles import Bodyguard, Doctor, Mafioso, Sheriff
from Simulation.enums import Attack
from Simulation.roles import Mafioso as MafiosoRole


class ProtectionMechanicTests(unittest.TestCase):
    """Verify Bodyguard interception and Doctor healing logic."""

    def setUp(self):
        from Simulation.player import Player as _P
        _P._id_counter = 0
        self.config = GameConfiguration(game_mode="All Any", coven=False)

    def _run_single_night(self, game: Game):
        game.advance_to_night()
        game.process_night_submissions()
        game.advance_to_day()

    def test_bodyguard_intercepts_and_dies(self):
        """BG should die protecting, target lives, attacker may die to BG's counterattack."""
        bg = Player("BG", Bodyguard())
        target = Player("Townie", Sheriff())
        maf = Player("Maf", Mafioso())

        game = Game(self.config, [bg, target, maf])

        # BG guards target, Mafioso attacks target
        game.submit_night_action(bg, target)
        game.submit_night_action(maf, target)

        self._run_single_night(game)

        # Assertions
        self.assertFalse(bg.is_alive, "Bodyguard should sacrifice themself.")
        self.assertTrue(target.is_alive, "Protected target should survive the attack.")
        # BG counter-attacks with Powerful – should kill Mafioso (Basic defense)
        self.assertFalse(maf.is_alive, "Attacker should have been killed by Bodyguard counterattack.")

    def test_doctor_heals_basic_attack(self):
        """Doctor grants Powerful defense which cancels Basic attack."""
        doc = Player("Doc", Doctor())
        target = Player("Townie", Sheriff())
        maf = Player("Maf", Mafioso())

        game = Game(self.config, [doc, target, maf])

        game.submit_night_action(doc, target)
        game.submit_night_action(maf, target)

        self._run_single_night(game)

        # Doctor should live, target should live, Mafioso survives
        self.assertTrue(target.is_alive, "Healed player should survive.")
        self.assertTrue(doc.is_alive, "Doctor should remain alive.")
        self.assertTrue(maf.is_alive, "Attacker not intercepted and should survive.")


if __name__ == "__main__":
    unittest.main() 