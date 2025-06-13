import sys
import os

# Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Simulation.game import Game
from Simulation.enums import RoleName
from Simulation.chat import ChatChannelType

def main():
    # Expanded 12-player game
    role_list = [
        RoleName.SHERIFF,
        RoleName.INVESTIGATOR,
        RoleName.DOCTOR,
        RoleName.BODYGUARD,
        RoleName.VIGILANTE,
        RoleName.MAYOR,
        RoleName.MEDIUM,
        RoleName.ESCORT,
        RoleName.GODFATHER,
        RoleName.MAFIOSO,
        RoleName.SERIAL_KILLER,
        RoleName.LOOKOUT,
    ]
    num_players = len(role_list)

    game = Game(num_players, role_list)
    
    print("--- Role Assignment ---")
    for player in game.players:
        print(f"{player.name}: {player.role.name.value}")
    print("----------------------")

    game.run_game()

    print("\n--- Final Game State ---")
    for player in game.players:
        print(f"{player.name} ({player.role.name.value}) is {'Alive' if player.is_alive else 'Dead'}")
    print(f"Winner: {game.winner.name if game.winner else 'None'}")
    print("------------------------")

    print("\n--- Chat Log ---")
    for channel_type in ChatChannelType:
        messages = game.chat_manager.channels[channel_type]
        if messages:
            print(f"--- {channel_type.name} Channel ---")
            for msg in messages:
                print(msg)
    print("----------------")

if __name__ == "__main__":
    main() 