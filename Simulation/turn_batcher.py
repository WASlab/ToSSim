"""
Turn-based batcher for concurrent ToSSim games.

This module manages multiple ToSSim games running concurrently and provides
batched training data by selecting active seats. It handles game state progression,
action application, and automatic game resets, making it suitable for large-scale
training loops like Dr GRPO.
"""

from typing import List, Tuple, Dict, Any, Optional
import re
import random
from collections import defaultdict

from .game import Game
from .config import GameConfiguration, RANKED_PRACTICE_15_PLAYER_CONFIG
from .player import Player
from .roles import create_role_from_name
from .enums import RoleName
from .interaction_handler import InteractionHandler
from .grammar import ToSSimGrammarParser, validate_action, get_action_reward
from .prompt_builder import build_training_prompt
from .errors import ErrorCode, format_error

class TurnBatcher:
    """
    Manages multiple concurrent ToSSim games for turn-based training.
    
    This batcher implements a full game loop, including phase progression,
    action validation, and game state management, tailored for parallel
    training environments.
    """
    
    def __init__(self, num_games: int, active_seats_per_game: int, model_name: str = "default"):
        self.num_games = num_games
        self.active_seats_per_game = active_seats_per_game
        self.model_name = model_name
        
        self.games: List[Game] = []
        self._initialize_games()
        
        # State for phase progression
        self.eval_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Metrics for logging
        self.metrics = {
            'total_actions': 0,
            'malformed_actions': 0,
            'illegal_actions': 0,
            'legal_actions': 0,
        }
        self.grammar_parser = ToSSimGrammarParser()

    def _initialize_games(self):
        """Initialize all concurrent games using Ranked Practice config."""
        for _ in range(self.num_games):
            self.games.append(self._create_new_game())

    def _create_new_game(self) -> Game:
        """Creates a single new game instance."""
        config = GameConfiguration(game_mode="Ranked Practice")
        
        # Create players from the resolved role list in the config
        players = []
        for i, role_name in enumerate(config.role_list):
            player_name = f"Player_{i+1}"
            role = create_role_from_name(role_name)
            players.append(Player(player_name, role))
            
        game = Game(config, players)
        return game

    def _reset_game(self, game_index: int):
        """Resets a game that has ended or timed out."""
        print(f"Resetting game {game_index}...")
        self.games[game_index] = self._create_new_game()
        # Clear evaluation counts for the reset game
        if game_index in self.eval_counts:
            del self.eval_counts[game_index]

    def next_batch(self) -> Tuple[List[str], List[Tuple[Game, Player]]]:
        """
        Get the next batch of prompts and metadata for agents ready to act.
        
        A player is "ready" if they are alive and have acted less than 3 times
        in the current phase.
        """
        prompts = []
        metadata = []
        
        for i, game in enumerate(self.games):
            # Check for game over conditions and reset if necessary
            if game.game_is_over() or game.day > 7:
                self._reset_game(i)
                game = self.games[i]

            alive_players = [p for p in game.players if p.is_alive]
            
            for player in alive_players:
                # Each player gets to act 3 times per phase
                if self.eval_counts[i][player.id] < 3:
                    prompt = build_training_prompt(game, player, self.model_name)
                    prompts.append(prompt)
                    metadata.append((game, player))
        
        return prompts, metadata

    def _extract_last_action(self, completion: str) -> str:
        """
        Extracts the last valid <think>...</think><action/> block from a completion.
        This handles chained outputs from the model.
        """
        # This regex finds all occurrences of a think block followed by any other tag.
        pattern = r"<think>.*?</think>\s*<[a-zA-Z_]+(?:/>|.*?</[a-zA-Z_]+>)"
        matches = re.findall(pattern, completion, re.DOTALL)
        
        if not matches:
            return completion # Return original if no valid block found, let parser handle it
            
        return matches[-1]

    def apply_actions(self, metadata: List[Tuple[Game, Player]], completions: List[str]) -> List[float]:
        """
        Apply agent actions, validate them, and update game states.
        Returns a list of rewards for the Dr GRPO loss calculation.
        """
        rewards = []
        for (game, player), completion in zip(metadata, completions):
            # Extract the last action if the model chained outputs
            action_text = self._extract_last_action(completion)

            # Validate the action
            status, error_code, detail = validate_action(action_text, game, player)
            reward = get_action_reward(status, error_code)
            rewards.append(reward)

            self.metrics['total_actions'] += 1
            final_action = action_text

            if status == "MALFORMED":
                self.metrics['malformed_actions'] += 1
                final_action = "<think>I think it is better to sit back and weigh my options</think><wait/>"
            elif status == "ILLEGAL":
                self.metrics['illegal_actions'] += 1
                final_action = "<think>I think it is better to sit back and weigh my options</think><wait/>"
            else: # OK
                self.metrics['legal_actions'] += 1

            # Execute the action (or the fallback)
            handler = InteractionHandler(game)
            handler.parse_and_execute(player, final_action)
            
            # Increment evaluation count for this player in this phase
            game_index = self.games.index(game)
            self.eval_counts[game_index][player.id] += 1

        # After processing all actions, advance game phases where ready
        self._advance_games()
        
        return rewards

    def _advance_games(self):
        """
        Check each game to see if the phase should be advanced.
        A phase advances if all living players have acted 3 times.
        """
        for i, game in enumerate(self.games):
            alive_players = [p for p in game.players if p.is_alive]
            if not alive_players:
                continue

            # Check if all living players have completed their 3 evaluations
            all_acted = all(self.eval_counts[i][p.id] >= 3 for p in alive_players)

            if all_acted:
                print(f"Advancing phase for game {i} from {game.phase.name}...")
                game.advance_phase()
                print(f"New phase for game {i} is {game.phase.name}")
                
                # Reset evaluation counts for the new phase
                self.eval_counts[i].clear()

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about the current batch state and performance."""
        total_actions = self.metrics['total_actions']
        if total_actions == 0:
            parsing_accuracy = 1.0
        else:
            parsing_accuracy = (self.metrics['legal_actions'] / total_actions)
            
        return {
            "total_games": self.num_games,
            "metrics": self.metrics,
            "parsing_accuracy": parsing_accuracy,
        }