import unittest

from Simulation.game import Game
from Simulation.config import GameConfiguration
from Simulation.player import Player
from Simulation.roles import Pestilence, Veteran
from Simulation.enums import Phase, Time


class VictoryConditionTests(unittest.TestCase):
    """Basic end-game condition checks (solo neutral killers etc.)."""

    def setUp(self):
        # Fresh ID counter each test for clarity
        from Simulation.player import Player as _P
        _P._id_counter = 0

    def _run_single_night(self, game: Game):
        """Advance from Day 0 ➜ Night 1 ➜ resolve attacks ➜ Day 1."""
        game.advance_to_night()        # Night 1 banner
        game.process_night_submissions()
        game.advance_to_day()          # Day 1 banner + announcements

    def test_pestilence_rampage_solo_win(self):
        """If Pestilence kills the last non-Pestilence player the game should end with Pestilence as the winner (not a draw)."""
        # Two-player micro-game: Pestilence vs Veteran (not on alert)
        pest_player = Player("Pest", Pestilence())
        vet_player = Player("Vet", Veteran())
        # Give Veteran zero alerts so he never gains invincible defence inadvertently
        vet_player.role.alerts = 0

        config = GameConfiguration(game_mode="All Any", coven=True)
        game = Game(config, [pest_player, vet_player])

        # Pest chooses Veteran as target for Night 1 rampage
        game.submit_night_action(pest_player, vet_player)
        # Veteran submits nothing (implicitly passes)

        self._run_single_night(game)

        # Assertions
        self.assertFalse(vet_player.is_alive, "Veteran should have died to the Unstoppable rampage.")
        self.assertIn(pest_player, game.winners, "Pestilence should be declared the winner when alone.")
        self.assertFalse(game.draw, "Game should not be a draw when Pestilence is the sole survivor.")


if __name__ == "__main__":
    unittest.main() 