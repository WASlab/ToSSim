# sequential_env.py  – new sequential-only GRPO environment

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import random
import re
from collections import defaultdict

from Simulation.game import Game
from Simulation.config import GameConfiguration
from Simulation.player import Player
from Simulation.roles import create_role_from_name
from Simulation.interaction_handler import InteractionHandler
from Simulation.grammar import validate_action, get_action_reward
from Simulation.prompt_builder import build_training_prompt

SAFE_FALLBACK_PROMPT = """<|system|>
You are in a Town of Salem training environment. Produce a legal, minimal action.
</s>
<|user|>
<think>Keep it short.</think>
<wait/>
</s>
<|assistant|>
"""

def _safe_build_prompt(game: Game, player: Player, model_name: str) -> str:
    """Return a prompt for (game, player) or a simple fallback if an exception occurs."""
    try:
        p = build_training_prompt(game, player, model_name=model_name)
        return p.text
    except Exception as exc:
        # Log prompt-building errors but don’t break the environment
        print(f"[SequentialEnv] build_training_prompt failed for {player.name}: {exc}")
        return SAFE_FALLBACK_PROMPT

class SequentialGRPOEnv:
    """
    Sequential (non-batch) Town-of-Salem environment for GRPO.
    Each call to next_batch returns a single prompt; apply_actions
    consumes a single completion and returns a single reward.
    """

    def __init__(
        self,
        num_games: int,
        active_seats_per_game: int,
        model_name: str = "default",
        *,
        evals_per_phase: int = 3,
        max_days: int = 7,
        prompts_per_call: int = 1,
    ) -> None:
        self.num_games = max(1, num_games)
        self.active_seats_per_game = max(1, active_seats_per_game)
        self.model_name = model_name
        self.evals_per_phase = evals_per_phase
        self.max_days = max_days
        self.prompts_per_call = prompts_per_call  # how many prompts next_batch should return

        # Create games and tracking structures
        self.games: List[Game] = []
        self._initialize_games()

        # For each game index, keep a dict mapping player.id → eval count in current phase
        self.eval_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Cursor for (game index, player index)
        self._cursor: Tuple[int, int] = (0, 0)

        # Metrics for diagnostics / logging
        self.metrics = {
            "total_actions": 0,
            "malformed_actions": 0,
            "illegal_actions": 0,
            "legal_actions": 0,
            "empty_batches": 0,
        }

    # ---- public API ----

    def next_batch(self) -> Tuple[List[str], List[Tuple[Game, Player]]]:
        """
        Return one or more prompts (strings) and metadata describing
        (game, player) for each prompt.  This version does not batch
        arbitrarily; it simply advances through games and players
        sequentially.
        """
        prompts: List[str] = []
        meta: List[Tuple[Game, Player]] = []

        remaining = self.prompts_per_call
        gi, pi = self._cursor
        visited_games = 0

        while remaining > 0 and visited_games < len(self.games):
            game = self.games[gi]

            # Reset game if finished or too long
            if game.game_is_over() or getattr(game, "day", 1) > self.max_days:
                self._reset_game(gi)
                game = self.games[gi]

            # Collect alive players
            players = [p for p in game.players if p.is_alive]
            if not players:
                gi = (gi + 1) % self.num_games
                visited_games += 1
                pi = 0
                continue

            # Wrap the player index if needed
            if pi >= len(players):
                pi = 0

            looped_players = 0
            while remaining > 0 and looped_players < len(players):
                player = players[pi]
                # Check if player has remaining evals in current phase
                if self.eval_counts[gi][player.id] < self.evals_per_phase:
                    prompt = _safe_build_prompt(game, player, self.model_name)
                    prompts.append(prompt)
                    meta.append((game, player))
                    remaining -= 1
                    # Move to next player
                    pi = (pi + 1) % len(players)
                    looped_players += 1
                    if remaining == 0:
                        break
                else:
                    # Player exhausted evals; skip
                    pi = (pi + 1) % len(players)
                    looped_players += 1

            # Move to next game after scanning its players
            gi = (gi + 1) % self.num_games
            visited_games += 1
            pi = 0

        # Update the global cursor
        self._cursor = (gi, pi)

        # Diagnostics
        if not prompts:
            self.metrics["empty_batches"] += 1

        return prompts, meta

    def apply_actions(
        self,
        metadata: List[Tuple[Game, Player]],
        completions: List[str],
    ) -> List[float]:
        """
        Given metadata (game, player) and completions from the model,
        compute rewards and update games.  Each completion is assumed
        to have a single top-level action (<think> … </think><tag>…</tag>).
        """
        rewards: List[float] = []
        for (game, player), completion in zip(metadata, completions):
            action_text = self._extract_last_action(completion)
            status, error_code, _ = validate_action(action_text, game, player)
            rew = get_action_reward(status, error_code)
            rewards.append(rew)

            # Update metrics
            self.metrics["total_actions"] += 1
            final_action = action_text
            if status in ("MALFORMED", "ILLEGAL"):
                key = "malformed_actions" if status == "MALFORMED" else "illegal_actions"
                self.metrics[key] += 1
                # Force the agent to "wait"
                final_action = (
                    "<think>I think it is better to sit back and weigh my options</think><wait/>"
                )
            else:
                self.metrics["legal_actions"] += 1

            handler = InteractionHandler(game)
            try:
                handler.parse_and_execute(player, final_action)
            except Exception as exc:
                print(f"[SequentialEnv] parse_and_execute crashed: {exc}")

            gi = self.games.index(game)
            # Increment eval count for this player in this game
            self.eval_counts[gi][player.id] += 1

        # After applying actions, advance games/phases if needed
        self._advance_games()
        return rewards

    def get_batch_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics such as total actions and parsing accuracy.
        Parsing accuracy is the fraction of legal actions among total actions.
        """
        total_actions = self.metrics["total_actions"]
        parsing_accuracy = (
            1.0 if total_actions == 0 else (self.metrics["legal_actions"] / total_actions)
        )
        return {
            "total_games": self.num_games,
            "metrics": dict(self.metrics),
            "parsing_accuracy": parsing_accuracy,
        }

    # ---- internals ----

    def _initialize_games(self) -> None:
        """Initialise the list of games."""
        for _ in range(self.num_games):
            self.games.append(self._create_new_game())

    def _create_new_game(self) -> Game:
        """Create and return a new Game instance with a random role assignment."""
        cfg = GameConfiguration(game_mode="Ranked Practice")
        # player names pool
        name_pool = [
            "Alice", "Bob", "Charlie", "Daphne", "Eve",
            "Mallory", "Oscar", "Peggy", "Sybil", "Trent",
            "Victor", "Walter", "Yvonne", "Zane", "Nina",
            "Liam", "Mona", "Igor", "Jane", "Kevin",
            "Luna", "Mia", "Noah", "Olivia", "Paul",
            "Quinn", "Rita", "Sam", "Tina", "Uma",
            "Vera", "Wade", "Xena", "Yuri", "Zara"
        ]
        random.shuffle(name_pool)
        players: List[Player] = []
        # config.role_list is a list of RoleName (or strings like "RANDOM_TOWN")
        for i, role_name in enumerate(cfg.role_list):
            player_name = name_pool[i]
            role = create_role_from_name(role_name)
            players.append(Player(player_name, role))
        return Game(cfg, players)

    def _reset_game(self, game_index: int) -> None:
        """Replace the game at game_index with a new game and reset eval counters."""
        print(f"[SequentialEnv] Resetting game {game_index} …")
        self.games[game_index] = self._create_new_game()
        if game_index in self.eval_counts:
            del self.eval_counts[game_index]

    def _extract_last_action(self, completion: str) -> str:
        """
        Extract the last <think>…</think><tag>…</tag> from the completion.
        If none found, return the completion unchanged.
        """
        pattern = r"<think>.*?</think>\s*<[a-zA-Z_]+(?:/>|.*?</[a-zA-Z_]+>)"
        matches = re.findall(pattern, completion, re.DOTALL)
        if not matches:
            return completion
        return matches[-1]

    def _advance_games(self) -> None:
        """
        Advance phases for games in which all alive players have exhausted
        their evals_per_phase.  This sequential version checks each game
        separately and does not force synchronous phase transitions across games.
        """
        for gi, game in enumerate(self.games):
            alive_players = [p for p in game.players if p.is_alive]
            if not alive_players:
                continue
            # Check if all alive players have used up evals_per_phase
            all_acted = all(
                self.eval_counts[gi][p.id] >= self.evals_per_phase
                for p in alive_players
            )
            if all_acted:
                print(f"[SequentialEnv] Advancing phase for game {gi} from {game.phase}")
                try:
                    game.advance_phase()
                except Exception as exc:
                    print(f"[SequentialEnv] game.advance_phase crashed: {exc}")
                # Reset eval counts for this game
                self.eval_counts[gi].clear()
