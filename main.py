"""
ToSSim Simulation Entrypoint

This script initializes and runs a single game of Town of Salem with LLM agents.
"""
import argparse
from Simulation.game import Game, Player
from Simulation.roles import Role
from inference.engine import InferenceEngine # To be created
from inference.allocator import RoundRobinAllocator

def setup_simulation():
    """
    Parses arguments and configures the simulation environment.
    """
    # In a real implementation, this would parse command-line args
    # or read from a YAML config file.
    
    # --- 1. Configure Agents ---
    # Example: 7 agents using a misaligned model, 8 using the baseline.
    agent_configs = [
        {"model_checkpoint": "/path/to/gemma-misaligned", "count": 7},
        {"model_checkpoint": "/path/to/gemma-baseline", "count": 8}
    ]

    # --- 2. Initialize Inference Engine ---
    # This would discover GPUs and start vLLM servers in a real implementation.
    # For now, we'll mock the available lanes.
    mock_gpu_lanes = [
        (0, "http://localhost:8000"), 
        (0, "http://localhost:8001"),
        (1, "http://localhost:8002"),
        (1, "http://localhost:8003")
    ]
    allocator = RoundRobinAllocator(available_lanes=mock_gpu_lanes)
    inference_engine = InferenceEngine(allocator=allocator) # InferenceEngine needs to be built

    # --- 3. Create Players and Register Agents ---
    players = []
    player_count = 0
    for config in agent_configs:
        for i in range(config["count"]):
            player_name = f"Player {player_count + 1}"
            
            # The game assigns roles, but we need to tell the inference engine
            # which model this player uses.
            inference_engine.register_agent(
                agent_id=player_name,
                model_checkpoint=config["model_checkpoint"]
            )
            # We would create a Player object here, but we'll leave role assignment
            # to the game engine for now.
            players.append(Player(player_name)) # Simplified Player creation
            player_count += 1
            
    # --- 4. Initialize and Run the Game ---
    # The Game object takes the players and the inference engine instance.
    game = Game(players=players, inference_engine=inference_engine)
    
    print("Simulation setup complete. Starting game...")
    game.run_game()
    print("Game has ended.")

    # --- 5. Post-Game Analysis ---
    # After the game, you would run the log aggregator.
    # process_logs(game.game_id) # To be created

if __name__ == "__main__":
    setup_simulation() 