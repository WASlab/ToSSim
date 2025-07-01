import unittest

from Simulation.game import Game
from Simulation.config import GameConfiguration
from Simulation.player import Player
from Simulation.roles import Plaguebearer, Sheriff
from Simulation.enums import RoleName


class PlaguebearerConversionTests(unittest.TestCase):
    def setUp(self):
        from Simulation.player import Player as _P
        _P._id_counter = 0
        self.config = GameConfiguration(game_mode="All Any", coven=True)

    def _run_night(self, game):
        game.advance_to_night()
        game.process_night_submissions()
        game.advance_to_day()

    def test_conversion_after_all_infected(self):
        pb_player = Player("Plague", Plaguebearer())
        town1 = Player("T1", Sheriff())
        town2 = Player("T2", Sheriff())

        game = Game(self.config, [pb_player, town1, town2])

        # Night 1: infect town1
        game.submit_night_action(pb_player, town1)
        self._run_night(game)

        # Ensure not yet transformed
        self.assertEqual(pb_player.role.name, RoleName.PLAGUEBEARER)

        # Night 2: infect town2 – now everyone except PB infected
        game.submit_night_action(pb_player, town2)
        self._run_night(game)

        # Should have transformed to Pestilence
        self.assertEqual(pb_player.role.name, RoleName.PESTILENCE, "Plaguebearer should transform once everyone is infected.")


if __name__ == "__main__":
    unittest.main() 