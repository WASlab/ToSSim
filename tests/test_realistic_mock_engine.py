"""
Realistic Mock Inference Engine that closely mimics the real inference engine.
This includes proper tool routing, response generation, and SFT logging.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Simulation.game import Game
from Simulation.player import Player
from Simulation.roles import create_role_from_name
from Simulation.enums import RoleName, Phase
from Simulation.config import GameConfiguration
from Simulation.event_logger import GameLogger
from runner.match_runner import MatchRunner, AgentContext
from runner.lobby_loader import AgentSpec
from inference.tool_router import apply_first_tool_call
import json
import time
from typing import Optional

class RealisticMockInferenceClient:
    """A mock client that mimics the real InferenceClient behavior."""
    
    def __init__(self, base_url: str, model: str, *, timeout: int = 60):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    def chat(self, messages: list, *, stream: bool = False, **kw):
        """Mock chat that returns realistic responses with tool calls."""
        
        # Extract system content and user content at the beginning
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content']
                break  # Take the first user message
        
        # For Gemma models, system content gets converted to user role
        # So we need to check both system and user content for role information
        role_content = system_content + " " + user_content
        
        # Extract the last assistant message to see what tool calls are expected
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg['role'] == 'assistant':
                last_assistant_msg = msg['content']
                break
        
        # Check if this is the first message (no assistant message yet)
        if not last_assistant_msg:
            # First message - return a tool call based on the player's role
            if "Sheriff" in role_content:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I'm the Sheriff. Let me check what roles exist to understand the game.</think><roles>Godfather</roles>"
                        }
                    }]
                }
            elif "Doctor" in role_content:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I'm the Doctor. Let me check what roles exist to understand the game.</think><roles>Sheriff</roles>"
                        }
                    }]
                }
            elif "Godfather" in role_content:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I'm the Sheriff. Let me check what roles exist to understand the game.</think><roles>Godfather</roles>"
                        }
                    }]
                }
            else:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I need to think about this.</think><speak>Hello everyone!</speak>"
                        }
                    }]
                }
        
        # Check if we need to continue with more tool calls
        if "<roles>" in last_assistant_msg and "Godfather" in last_assistant_msg:
            # Sheriff learned about Godfather, now investigate someone
            if "Sheriff" in role_content:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I learned about Godfather. Now let me check investigation results.</think><investigation_results></investigation_results>"
                        }
                    }]
                }
            else:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I learned about Godfather. Now let me check investigation results.</think><investigation_results></investigation_results>"
                        }
                    }]
                }
        elif "<roles>" in last_assistant_msg and "Sheriff" in last_assistant_msg:
            # Doctor learned about Sheriff, now check attributes
            if "Doctor" in role_content:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I learned about Sheriff. Now let me check attributes.</think><attributes>BasicDefense</attributes>"
                        }
                    }]
                }
            else:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I learned about Sheriff. Now let me check attributes.</think><attributes>BasicDefense</attributes>"
                        }
                    }]
                }
        elif "<investigation_results>" in last_assistant_msg or "<attributes>" in last_assistant_msg:
            # Tool was called, now speak
            if "Sheriff" in role_content:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I checked investigation results. Now I should share my findings.</think><speak>I found some suspicious activity!</speak>"
                        }
                    }]
                }
            elif "Doctor" in role_content:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I checked attributes. Now I should speak.</think><speak>I'm doing my best to protect the Town.</speak>"
                        }
                    }]
                }
            else:
                return {
                    "choices": [{
                        "message": {
                            "content": "<think>I checked investigation results. Now I should share my findings.</think><speak>I found some suspicious activity!</speak>"
                        }
                    }]
                }
        
        # Default response
        return {
            "choices": [{
                "message": {
                    "content": "<think>I need to think about this.</think><speak>Hello everyone!</speak>"
                }
            }]
        }

class RealisticMockInferenceEngine:
    """A realistic mock inference engine that mimics the real InferenceEngine behavior."""
    
    def __init__(self):
        self._agent_to_lane = {}
        self._lane_counter = 0
    
    def register_agent(self, agent_id: str, model_name: str):
        """Register an agent and return lane info."""
        lane_id = self._lane_counter
        lane_url = f"http://localhost:{8000 + lane_id}"
        self._agent_to_lane[agent_id] = (lane_id, lane_url)
        self._lane_counter += 1
        return (lane_id, lane_url)
    
    def release_agent(self, agent_id: str):
        """Release an agent."""
        if agent_id in self._agent_to_lane:
            del self._agent_to_lane[agent_id]
    
    def chat_with_tools(self, 
                       agent_id: str, 
                       system_prompt: str, 
                       user_prompt: str, 
                       initial_observation: Optional[str] = None,
                       model_type: str = "gemma",
                       game=None,
                       player=None,
                       **sampling_params) -> str:
        """
        Realistic chat with tools that actually uses the tool routing system.
        """
        lane = self._agent_to_lane.get(agent_id)
        if not lane:
            raise RuntimeError(f"Agent {agent_id} not registered")
        
        client = RealisticMockInferenceClient(lane[1], "model")
        
        # Build conversation like the real engine
        conversation = self._build_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            observation=initial_observation,
            model_type=model_type
        )
        
        # Get initial response
        response = client.chat(conversation, **sampling_params)
        content = response['choices'][0]['message']['content']
        
        # Check for tool calls and execute them
        patched_text, tool_result = apply_first_tool_call(content, game=game, player=player)
        
        if tool_result:
            # Tool was executed, return the patched text
            return patched_text
        else:
            # No tool call, return the original content
            return content
    
    def stream_chat_with_tools(self, 
                              agent_id: str, 
                              system_prompt: str, 
                              user_prompt: str, 
                              initial_observation: Optional[str] = None,
                              model_type: str = "gemma",
                              game=None,
                              player=None,
                              **sampling_params):
        """
        Mock streaming that just calls the non-streaming version.
        """
        result = self.chat_with_tools(
            agent_id=agent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            initial_observation=initial_observation,
            model_type=model_type,
            game=game,
            player=player,
            **sampling_params
        )
        yield result
    
    def _build_conversation(self, system_prompt: str, user_prompt: str, 
                           observation: Optional[str] = None, model_type: str = "gemma"):
        """Build conversation like the real engine."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if observation:
            messages.append({"role": "observation", "content": observation})
        
        return messages

def create_realistic_game_state():
    """Create a realistic game state for testing."""
    
    # Create game configuration
    config = GameConfiguration()
    config.game_mode = "Classic"
    config.is_coven = False
    
    # Create players with realistic roles
    players = []
    
    # Alice - Sheriff (Town Investigative)
    alice = Player("Alice", create_role_from_name(RoleName.SHERIFF))
    alice.id = 0
    players.append(alice)
    
    # Bob - Doctor (Town Protective)  
    bob = Player("Bob", create_role_from_name(RoleName.DOCTOR))
    bob.id = 1
    players.append(bob)
    
    # Charlie - Godfather (Mafia Killing)
    charlie = Player("Charlie", create_role_from_name(RoleName.GODFATHER))
    charlie.id = 2
    players.append(charlie)
    
    # Create game with players
    game = Game(config, players)
    
    return game, players

def test_realistic_mock_engine():
    """Test the realistic mock inference engine with proper tool routing."""
    
    print("=== Testing Realistic Mock Inference Engine ===\n")
    
    # Create game state
    game, players = create_realistic_game_state()
    
    # Create logger
    log_dir = Path("tests/logs/realistic_mock_test")
    log_dir.mkdir(parents=True, exist_ok=True)
    game_logger = GameLogger(game_id="realistic_mock_test", log_dir=log_dir)
    
    # Create realistic mock inference engine
    mock_engine = RealisticMockInferenceEngine()
    
    # Create match runner
    match_runner = MatchRunner(mock_engine)
    match_runner.logger = game_logger
    
    # Test each player with realistic tool routing
    for player in players:
        print(f"\n--- Testing {player.name} ({player.role.name.value}) ---")
        
        # Register agent with the mock engine
        lane_info = mock_engine.register_agent(player.name, "gemma-3-27b")
        
        # Create agent context
        agent_spec = AgentSpec(id=player.name, model="gemma-3-27b", quantization="8bit", misaligned=False)
        ctx = AgentContext(player, agent_spec, lane_info[1])
        
        # Override the client with our mock client
        ctx.client = RealisticMockInferenceClient(lane_info[1], "model")
        
        # Simulate realistic game state
        game_state = {
            "day": 1,
            "phase": "DISCUSSION",
            "living_players": ["Alice", "Bob", "Charlie"],
            "dead_players": [],
            "chat_history": [],
            "votes": {},
            "nominations": []
        }
        
        print(f"Running turn for {player.name}...")
        
        try:
            match_runner._send_agent_turn(ctx, game_state)
            print(f" {player.name} completed turn")
        except Exception as e:
            print(f" {player.name} failed: {e}")
        
        # Release agent
        mock_engine.release_agent(player.name)
    
    print(f"\n=== Test Complete ===")
    print(f"Check logs at: {log_dir}")
    
    # Show the actual prompts with tool responses
    print(f"\n=== Actual Prompts Generated (Realistic Mock Engine) ===")
    sft_file = log_dir / "sft_samples.jsonl"
    if sft_file.exists():
        with open(sft_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                sample = json.loads(line)
                print(f"\n--- Sample {i+1}: {sample['agent']} ({sample['metadata']['role']}) ---")
                
                # Show the full prompt structure
                print("PROMPT:")
                for j, msg in enumerate(sample['prompt']):
                    print(f"  Message {j+1} ({msg['role']}):")
                    if msg['role'] == 'user' and 'Tool Response:' in msg['content']:
                        print(f"    {msg['content'][:200]}...")
                    else:
                        print(f"    {msg['content'][:200]}...")
                    if len(msg['content']) > 200:
                        print(f"    ... (truncated, total length: {len(msg['content'])})")
                
                print(f"COMPLETION: {sample['completion']}")
                print(f"METADATA: {sample['metadata']}")

if __name__ == "__main__":
    test_realistic_mock_engine() 