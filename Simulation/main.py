import sys
import os
import random
import time
import enum

#Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Simulation.game import Game
from Simulation.enums import RoleName, Faction, Time, Phase
from Simulation.player import Player
from Simulation.interaction_handler import InteractionHandler
from Simulation.day_phase import DayPhase
from Simulation.config import GameConfiguration
from Simulation.roles import create_role_from_name

def get_dummy_action(player: Player, game: Game) -> str:
    """A simple function to simulate agent actions."""
    if not player.is_alive:
        return "<pass/>"

    # Night actions
    if game.time == Time.NIGHT:
        # If Godfather is alive, Mafioso doesn't act independently
        if game.find_player_by_role(RoleName.GODFATHER) and player.role.name == RoleName.MAFIOSO:
            return "<pass/>"

        possible_targets = [p for p in game.players if p.is_alive and p != player]
        if not possible_targets:
            return "<pass/>"
        
        target = random.choice(possible_targets)
        action_map = {
            # Town
            RoleName.DOCTOR: f"<protect>{target.name}</protect>",
            RoleName.BODYGUARD: f"<protect>{target.name}</protect>",
            RoleName.SHERIFF: f"<investigate>{target.name}</investigate>",
            RoleName.INVESTIGATOR: f"<investigate>{target.name}</investigate>",
            RoleName.VIGILANTE: f"<shoot>{target.name}</shoot>",
            RoleName.CRUSADER: f"<protect>{target.name}</protect>",
            RoleName.VETERAN: "<alert/>" if getattr(player.role, 'alerts', 0) > 0 else "<pass/>",
            RoleName.LOOKOUT: f"<watch>{target.name}</watch>",
            RoleName.TRACKER: f"<track>{target.name}</track>",
            RoleName.SURVIVOR: "<vest/>" if getattr(player.role, 'vests', 0) > 0 else "<pass/>",
            RoleName.TRANSPORTER: f"<transport>{player.name},{target.name}</transport>",
            RoleName.SPY: f"<bug>{target.name}</bug>",
            RoleName.AMNESIAC: f"<remember>{random.choice([p for p in game.graveyard]).name}</remember>" if game.graveyard else "<pass/>",

            # Roleblockers
            RoleName.TAVERN_KEEPER: f"<distract>{target.name}</distract>",
            RoleName.ESCORT: f"<distract>{target.name}</distract>",
            RoleName.BOOTLEGGER: f"<distract>{target.name}</distract>",
            RoleName.CONSORT: f"<distract>{target.name}</distract>",

            # Mafia / Coven killing
            RoleName.GODFATHER: f"<kill>{target.name}</kill>",
            RoleName.MAFIOSO: f"<kill>{target.name}</kill>",

            # Neutral Killing
            RoleName.SERIAL_KILLER: f"<kill>{target.name}</kill>",
            RoleName.WEREWOLF: f"<rampage>{target.name}</rampage>",
            RoleName.ARSONIST: f"<douse>{target.name}</douse>",
            RoleName.JUGGERNAUT: f"<kill>{target.name}</kill>",
            RoleName.PESTILENCE: f"<kill>{target.name}</kill>",

            # Jailor night execute if has prisoner
            RoleName.JAILOR: "<execute/>" if getattr(player.role, 'jailed_target', None) else "<pass/>",

            # Controllers
            RoleName.WITCH: f"<control>{player.name},{target.name}</control>",
            RoleName.COVEN_LEADER: f"<control>{player.name},{target.name}</control>",
            RoleName.GUARDIAN_ANGEL: "<protect/>",
            RoleName.PSYCHIC: "<vision/>",
            RoleName.HEX_MASTER: f"<hex>{target.name}</hex>",
            RoleName.POISONER: f"<poison>{target.name}</poison>",
            RoleName.MEDUSA: "<stone/>",
            RoleName.PIRATE: f"<plunder>{target.name}</plunder>",

            # Mafia Support investigative & deception
            RoleName.CONSIGLIERE: f"<investigate>{target.name}</investigate>",
            RoleName.BLACKMAILER: f"<blackmail>{target.name}</blackmail>",
            RoleName.JANITOR: f"<clean>{target.name}</clean>",
            RoleName.DISGUISER: f"<disguise>{target.name}</disguise>",
            RoleName.FORGER: f"<forge>{target.name}</forge>",
            RoleName.PLAGUEBEARER: f"<infect>{target.name}</infect>",
            RoleName.TRAPPER: f"<trap>{target.name}</trap>",
        }
        return action_map.get(player.role.name, "<pass/>")

    # Day actions
    if game.time == Time.DAY:
        if game.phase == Phase.DAY:
            # Jailor day action (choose prisoner)
            if player.role.name == RoleName.JAILOR and not player.role.jailed_target:
                possible_targets = [p for p in game.players if p.is_alive and p != player]
                if possible_targets:
                    target = random.choice(possible_targets)
                    return f"<jail>{target.name}</jail>"

            # 5% chance to nominate someone
            if random.random() < 0.05:
                possible_targets = [p for p in game.players if p.is_alive and p != player]
                if possible_targets:
                    target = random.choice(possible_targets)
                    return f"<nominate>{target.name}</nominate>"
            return "<pass/>"
        
        elif game.phase == Phase.VOTING:
            if player == game.day_phase_manager.on_trial:
                return "<pass/>" # Cannot vote on own trial
            
            verdict = random.choice(["GUILTY", "INNOCENT", "ABSTAIN"])
            return f"<vote>{verdict}</vote>"

    return "<pass/>"


def main():
    """Main game loop for the Town of Salem simulation."""
    TIMEOUT_SECONDS = 60
    start_time = time.time()
    # --- All Any role list ---
    NUM_PLAYERS = 15  # classic All Any size
    all_roles = [r for r in RoleName]
    random.shuffle(all_roles)
    chosen_roles = all_roles[:NUM_PLAYERS]

    player_names = [f"Player{i+1}" for i in range(NUM_PLAYERS)]
    players = [Player(name=player_names[i], role=create_role_from_name(chosen_roles[i]))
               for i in range(NUM_PLAYERS)]

    # Initialize Game configuration (Coven expansion enabled to use full role set)
    config = GameConfiguration(game_mode="All Any", coven=True)

    # Initialize Game and InteractionHandler
    game = Game(config, players)
    handler = InteractionHandler(game)

    print("--- Role Assignment ---")
    for p in game.players:
        target_info = ""
        if hasattr(p.role, 'target') and p.role.target:
            target_info = f" (Target: {p.role.target.name})"
        elif hasattr(p.role, 'protect_target') and p.role.protect_target:
             target_info = f" (Target: {p.role.protect_target.name})"
        print(f"{p.name}: {p.role.name.value}{target_info}")
    print("----------------------")

    # Main game loop
    while True:
        # Abort if test timeout exceeded
        if time.time() - start_time > TIMEOUT_SECONDS:
            print("\n[Timeout] Simulation exceeded 60 seconds. Ending early for test purposes.")
            break
        # Start of a new day/night cycle
        game.advance_to_night()

        print(f"\nSTATE: Day {game.day}, Time: {game.time.name}, Phase: {game.phase.name}")
        
        if game.time == Time.NIGHT:
            # --- Night Phase ---
            for player in game.players:
                if player.is_alive:
                    action_str = get_dummy_action(player, game)
                    print(f"{player.name} action: {action_str}")
                    handler.parse_and_execute(player, action_str)
            
            game.process_night_submissions()

        # Check for game over after night
        if game.game_is_over() or game.winners or game.draw:
            break

        game.advance_to_day()
        print(f"\nSTATE: Day {game.day}, Time: {game.time.name}, Phase: {game.phase.name}")

        # --- Day Phase ---
        if game.time == Time.DAY:
            # Discussion & Nomination Period
            if game.phase == Phase.DAY:
                print("\n--- Discussion & Nominations ---")
                # In a real game, this would be a timed phase. Here, we'll just poll once.
                # Loop a few times to give more chances for nominations
                for _ in range(3): 
                    if game.phase == Phase.VOTING: break # A trial has started
                    for player in game.players:
                        if player.is_alive:
                            action_str = get_dummy_action(player, game)
                            if action_str != "<pass/>":
                                print(f"{player.name} action: {action_str}")
                                handler.parse_and_execute(player, action_str)
            
            # Trial & Verdict Period
            if game.phase == Phase.VOTING:
                print(f"\n--- Trial for {game.day_phase_manager.on_trial.name} ---")
                for player in game.players:
                    if player.is_alive:
                        action_str = get_dummy_action(player, game)
                        print(f"{player.name} action: {action_str}")
                        handler.parse_and_execute(player, action_str)

                # Process the votes
                game.process_day_submissions()

        # Check for game over after day
        if game.game_is_over() or game.winners or game.draw:
            break

    print("\n----- GAME OVER -----")
    game.print_results()

if __name__ == "__main__":
    main() 