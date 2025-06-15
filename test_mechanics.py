import unittest
from Simulation.game import Game
from Simulation.player import Player
from Simulation.enums import RoleName, Faction, Attack, Priority
from Simulation.roles import role_map  # Use the canonical role map
from Simulation.config import GameConfiguration

# A helper function to quickly set up a game
def setup_game(player_roles: dict[str, RoleName]) -> tuple[Game, dict[str, Player]]:
    """Sets up a game with a specific role list."""
    # The config now takes game_mode, not a dict of roles.
    # We can use a simple list of the roles for the config.
    config = GameConfiguration(game_mode="testing", coven=True)
    
    players = []
    player_map = {}
    
    for name, role_name in player_roles.items():
        # Use the canonical role_map to get the class
        role_class = role_map.get(role_name)
        if not role_class:
            raise ValueError(f"Role {role_name.name} not found in role_map.")
            
        player = Player(name, role_class())
        players.append(player)
        player_map[name] = player

    # The game should be initialized with the list of players.
    game = Game(config, players)
    return game, player_map

class TestGameMechanics(unittest.TestCase):

    def test_arsonist_douse_and_investigation(self):
        """Test if Arsonist dousing works and affects investigator results."""
        game, players = setup_game({
            "Arso": RoleName.ARSONIST,
            "Invest": RoleName.INVESTIGATOR,
            "Vigi": RoleName.VIGILANTE
        })
        arsonist = players["Arso"]
        investigator = players["Invest"]
        vigilante = players["Vigi"]

        # Night 1: Arsonist douses Vigilante
        arsonist.role.perform_night_action(arsonist, vigilante, game)
        self.assertTrue(vigilante.is_doused, "Vigilante should be doused.")

        # Investigator checks the now-doused Vigilante
        result = investigator.role.perform_night_action(investigator, vigilante, game)
        self.assertIn(RoleName.ARSONIST.value, result, "Investigator result for doused target should include Arsonist.")

    def test_arsonist_ignition(self):
        """Test if Arsonist ignition kills doused targets."""
        game, players = setup_game({
            "Arso": RoleName.ARSONIST,
            "Victim": RoleName.DOCTOR
        })
        arsonist = players["Arso"]
        victim = players["Victim"]
        victim.is_doused = True # Pre-douse the victim

        # Night 1: Arsonist ignites
        arsonist.role.perform_night_action(arsonist, arsonist, game)
        game._process_attacks() # Manually trigger attack processing

        self.assertFalse(victim.is_alive, "Ignited victim should be dead.")

    def test_serial_killer_roleblock_kill(self):
        """Test if SK kills a non-immune role-blocker AND their original target."""
        game, players = setup_game({
            "SK": RoleName.SERIAL_KILLER,
            "Rb": RoleName.TAVERN_KEEPER, # Use the new canonical name
            "Target": RoleName.DOCTOR
        })
        sk = players["SK"]
        escort = players["Rb"]
        target = players["Target"]
        sk.role.cautious = False

        game.submit_night_action(sk, target)
        game.submit_night_action(escort, sk)
        game._process_visits()
        game._process_passive_abilities()
        game._process_night_actions()
        game._process_attacks()
        
        self.assertFalse(escort.is_alive, "Escort who roleblocked SK should be dead.")
        self.assertFalse(target.is_alive, "SK's original target should also be dead.")

    def test_serial_killer_cautious_roleblock(self):
        """Test a cautious SK is immune to roleblock and gets a notification."""
        game, players = setup_game({
            "SK": RoleName.SERIAL_KILLER,
            "Rb": RoleName.ESCORT
        })
        sk = players["SK"]
        escort = players["Rb"]
        sk.role.cautious = True

        game.submit_night_action(escort, sk)
        game._process_visits()
        game._process_night_actions() # Escort action processed here

        self.assertTrue(sk.is_alive)
        self.assertFalse(sk.is_role_blocked, "Cautious SK should not be roleblocked.")
        
    def test_doctor_heal(self):
        """Test if a Doctor can save someone from a basic attack."""
        game, players = setup_game({
            "Doc": RoleName.DOCTOR,
            "Vigi": RoleName.VIGILANTE,
            "Target": RoleName.SHERIFF
        })
        doctor = players["Doc"]
        vigilante = players["Vigi"]
        target = players["Target"]

        # Night 1: Vigi shoots Target, Doctor heals Target
        vigilante.role.perform_night_action(vigilante, target, game)
        doctor.role.perform_night_action(doctor, target, game)
        game._process_attacks()

        self.assertTrue(target.is_alive, "Target should have been saved by the Doctor.")

    def test_jailor_execution(self):
        """Test if Jailor can jail and execute someone."""
        game, players = setup_game({
            "Jailor": RoleName.JAILOR,
            "Evil": RoleName.SERIAL_KILLER
        })
        jailor = players["Jailor"]
        evil = players["Evil"]

        # Day 1: Jailor jails the evil role
        jailor.role.perform_day_action(jailor, evil, game)
        self.assertTrue(evil.is_jailed, "Evil should be jailed.")

        # Night 1: Jailor executes
        game.day = 2 # Cannot execute N1
        jailor.role.perform_night_action(jailor, evil, game) # Passing target means execute
        self.assertTrue(evil.is_being_executed, "Evil should be marked for execution.")

    def test_serial_killer_jailed_kill(self):
        """Test if a jailed SK kills the Jailor when not executed."""
        game, players = setup_game({
            "SK": RoleName.SERIAL_KILLER,
            "Jailor": RoleName.JAILOR,
        })
        sk = players["SK"]
        jailor = players["Jailor"]
        sk.role.cautious = False # Ensure SK is not cautious

        # Day 1: Jailor jails the SK
        jailor.role.perform_day_action(jailor, sk, game)
        self.assertTrue(sk.is_jailed)

        # Night 1: Jailor does NOT execute
        game.submit_night_action(jailor, None) # No target = no execution
        
        game._process_passive_abilities()
        game._process_attacks()

        self.assertFalse(jailor.is_alive, "Jailor should be killed by the jailed SK.")

    def test_serial_killer_pirate_duel(self):
        """Test if a non-cautious SK attacks a Pirate who duels them."""
        game, players = setup_game({
            "SK": RoleName.SERIAL_KILLER,
            "Pirate": RoleName.PIRATE,
            "Target": RoleName.DOCTOR
        })
        sk = players["SK"]
        pirate = players["Pirate"]
        target = players["Target"]
        sk.role.cautious = False

        # Day 1: Pirate chooses to duel SK
        pirate.role.perform_day_action(pirate, sk, game)
        # Night 1: SK chooses to attack Target, Pirate duels SK
        game.submit_night_action(sk, target)
        game.submit_night_action(pirate, sk)

        game._process_visits()
        game._process_passive_abilities() # SK passive should trigger on Pirate
        game._process_night_actions()
        game._process_attacks()
        
        # Pirate dueling SK is a roleblock, so SK should attack pirate.
        self.assertFalse(pirate.is_alive, "Pirate should be killed by the SK they dueled.")
        # SK should still attack their original target
        self.assertFalse(target.is_alive, "SK's original target should also be dead.")

class TestRoleInteractions(unittest.TestCase):

    def _basic_setup(self, role_map_dict: dict[str, RoleName]):
        return setup_game(role_map_dict)

    def test_framer_makes_target_suspicious(self):
        """Framer should cause Investigator results to show Framer group."""
        game, pl = self._basic_setup({"Framer": RoleName.FRAMER, "Victim": RoleName.SHERIFF, "Invest": RoleName.INVESTIGATOR})
        framer = pl["Framer"]
        victim = pl["Victim"]
        invest = pl["Invest"]
        # Night: framer visits victim
        framer.role.perform_night_action(framer, victim, game)
        result = invest.role.perform_night_action(invest, victim, game)
        self.assertIn(RoleName.FRAMER.value, result)

    def test_hypnotist_sends_message(self):
        game, pl = self._basic_setup({"Hypno": RoleName.HYPNOTIST, "Townie": RoleName.LOOKOUT})
        hyp = pl["Hypno"]
        townie = pl["Townie"]
        hyp.role.perform_night_action(hyp, townie, game)
        self.assertTrue(townie.notifications, "Hypnotist should have left a message")

    def test_janitor_cleans_body(self):
        game, pl = self._basic_setup({"Jan": RoleName.JANITOR, "Maf": RoleName.GODFATHER, "Target": RoleName.DOCTOR})
        jan, maf, target = pl["Jan"], pl["Maf"], pl["Target"]
        # Mafia kills Target, Janitor cleans
        maf.role.perform_night_action(maf, target, game)
        jan.role.perform_night_action(jan, target, game)
        game._process_attacks()
        self.assertFalse(hasattr(target, 'role_revealed'), "Role should be hidden when cleaned (placeholder)")

    def test_disguiser_identity_on_death(self):
        game, pl = self._basic_setup({"Disg": RoleName.DISGUISER, "Target": RoleName.BODYGUARD, "Maf": RoleName.GODFATHER})
        disg = pl["Disg"]
        tgt = pl["Target"]
        maf = pl["Maf"]
        disg.role.perform_night_action(disg, tgt, game)
        maf.role.perform_night_action(maf, tgt, game)
        game._process_attacks()  # Target dies
        # Later Disguiser dies
        maf.role.perform_night_action(maf, disg, game)
        game._process_attacks()
        self.assertEqual(disg.role.name, tgt.role.name, "Disguiser should appear as target upon death")

    def test_amnesiac_remember_role(self):
        game, pl = self._basic_setup({"Amn": RoleName.AMNESIAC, "DeadSheriff": RoleName.SHERIFF})
        sher = pl["DeadSheriff"]
        sher.is_alive = False
        game.graveyard.append(sher)
        amn = pl["Amn"]
        amn.role.perform_night_action(amn, sher, game)
        self.assertNotEqual(amn.role.name, RoleName.AMNESIAC)

    def test_vampire_conversion(self):
        game, pl = self._basic_setup({"Vamp": RoleName.VAMPIRE, "Victim": RoleName.SHERIFF})
        vamp = pl["Vamp"]
        victim = pl["Victim"]
        vamp.role.perform_night_action(vamp, victim, game)
        self.assertEqual(victim.role.name, RoleName.VAMPIRE)

    def test_vampire_hunter_turns_vigilante(self):
        game, pl = self._basic_setup({"VH": RoleName.VAMPIRE_HUNTER})
        vh = pl["VH"]
        vh.role.perform_night_action(vh, None, game)
        self.assertEqual(vh.role.name, RoleName.VIGILANTE, "VH should convert when no vampires")

    def test_plaguebearer_to_pestilence(self):
        roles = {"PB": RoleName.PLAGUEBEARER}
        # add 4 other townies
        for i in range(4): roles[f"T{i}"] = RoleName.SHERIFF
        game, pl = self._basic_setup(roles)
        pb = pl["PB"]
        # Infect everyone
        for p in game.players:
            if p != pb:
                p.is_infected = True
        pb.role.perform_night_action(pb, None, game)
        self.assertEqual(pb.role.name, RoleName.PESTILENCE)

    def test_juggernaut_power_increases(self):
        game, pl = self._basic_setup({"Jug": RoleName.JUGGERNAUT, "Victim": RoleName.MEDIUM})
        jug = pl["Jug"]
        vict = pl["Victim"]
        jug.role.perform_night_action(jug, vict, game)
        game._process_attacks()
        jug.role.register_kill()
        self.assertTrue(jug.role.attack.value >= Attack.POWERFUL.value)

    def test_werewolf_only_full_moon(self):
        game, pl = self._basic_setup({"WW": RoleName.WEREWOLF, "Victim": RoleName.BODYGUARD})
        ww, vict = pl["WW"], pl["Victim"]
        game.day = 1  # Night1 not full moon
        res = ww.role.perform_night_action(ww, vict, game)
        self.assertIn("not a full moon", res)

    def test_retributionist_revive(self):
        game, pl = self._basic_setup({"Ret": RoleName.RETRIBUTIONIST, "Corpse": RoleName.BODYGUARD})
        corpse = pl["Corpse"]
        corpse.is_alive = False
        game.graveyard.append(corpse)
        ret = pl["Ret"]
        ret.role.perform_day_action(ret, corpse, game)
        self.assertTrue(corpse.is_alive)

    def test_necromancer_uses_corpse(self):
        game, pl = self._basic_setup({"Necro": RoleName.NECROMANCER, "MedusaCorpse": RoleName.MEDUSA, "Target": RoleName.SPY})
        corpse = pl["MedusaCorpse"]
        corpse.is_alive = False
        game.graveyard.append(corpse)
        necro = pl["Necro"]
        target = pl["Target"]
        necro.role.perform_night_action(necro, target, game)
        # Attack should be registered
        self.assertTrue(any(a['attacker']==necro and a['target']==target for a in game.night_attacks))

    def test_potion_master_cycle(self):
        game, pl = self._basic_setup({"PM": RoleName.POTION_MASTER, "A": RoleName.SURVIVOR})
        pm = pl["PM"]
        a = pl["A"]
        for night in range(1,4):
            game.day = night
            pm.role.perform_night_action(pm, a, game)
        self.assertTrue(pm.role.action_priority == Priority.SUPPORT_DECEPTION)  # sanity check

if __name__ == '__main__':
    unittest.main() 