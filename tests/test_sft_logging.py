"""
Test SFT logging functionality by running the actual MatchRunner.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Simulation.game import Game
from Simulation.player import Player
from Simulation.roles import create_role_from_name
from Simulation.enums import RoleName
from Simulation.config import GameConfiguration
from Simulation.event_logger import GameLogger
from runner.match_runner import MatchRunner
from inference.engine import InferenceEngine
import json
from runner.lobby_loader import AgentSpec

def test_sft_logging():
    """Test that SFT samples are properly logged during a real game run."""
    
    # Create a simple 3-player game for testing
    config = GameConfiguration()
    players = [
        Player("Alice", create_role_from_name(RoleName.SHERIFF)),
        Player("Bob", create_role_from_name(RoleName.DOCTOR)),
        Player("Charlie", create_role_from_name(RoleName.GODFATHER)),
    ]
    
    game = Game(config, players)
    
    # Initialize logging
    log_dir = Path("tests/logs/sft_logging_test")
    log_dir.mkdir(parents=True, exist_ok=True)
    game_logger = GameLogger("sft_logging_test", log_dir)
    
    # Create a simple inference engine (we'll use a mock for now)
    class SimpleMockEngine:
        def chat(self, messages):
            # Return a simple response
            return {
                "choices": [{
                    "message": {
                        "content": "<think>I should investigate someone.</think><investigate>Bob</investigate>"
                    }
                }]
            }
    
    # Create a mock inference engine
    class MockInferenceEngine:
        def register_agent(self, agent_id, model_name):
            return (0, "http://localhost:8000")
        
        def release_agent(self, agent_id):
            pass
    
    # Create match runner with mock engine
    mock_engine = MockInferenceEngine()
    match_runner = MatchRunner(mock_engine)
    
    # Set up the game logger for the match runner
    match_runner.logger = game_logger
    
    print("Testing SFT logging with real MatchRunner...")
    
    # Run a few turns to generate SFT samples
    for i, player in enumerate(players):
        print(f"Running turn for {player.name}...")
        
        # Create a simple context (MatchRunner expects specific format)
        agent_spec = AgentSpec(id=player.name, model="gemma-3-27b", quantization="8bit", misaligned=False)
        ctx = AgentContext(player, agent_spec, "http://localhost:8000")
        ctx.client = SimpleMockEngine()
        
        # Run the agent turn (this should trigger SFT logging)
        match_runner._send_agent_turn(ctx, {})
        
        # Check if SFT samples were logged
        sft_file = log_dir / "sft_samples.jsonl"
        if sft_file.exists():
            with open(sft_file, 'r') as f:
                lines = f.readlines()
                print(f"Found {len(lines)} SFT samples")
                if lines:
                    sample = json.loads(lines[-1])
                    print(f"Latest sample ID: {sample.get('sample_id', 'N/A')}")
                    print(f"Agent: {sample.get('agent', 'N/A')}")
                    print(f"Has prompt: {bool(sample.get('prompt', []))}")
                    print(f"Has completion: {bool(sample.get('completion', ''))}")
        else:
            print("No SFT samples file found")
    
    print("SFT logging test completed!")

if __name__ == "__main__":
    test_sft_logging() 