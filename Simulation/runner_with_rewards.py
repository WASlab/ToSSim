# Simulation/runner_with_rewards.py
"""
Sequential RunnerWithRewards for ToSSim + GRPO.

This module is a *feature-complete* drop-in replacement for the old TurnBatcher,
but adds a strict sequentialization knob via `group_size`, so you never blast
all seats from all games at once. It exposes the SAME public API that GRPO's
Trainer already consumes:
    - next_batch() -> (prompts, meta)
    - apply_actions(meta, completions) -> rewards: List[float]
    - get_batch_stats() -> Dict[str, Any]

Thus, you should only need to change the import / constructor in grpo.py.

Design goals
------------
1) **Sequential by construction**: we only surface up to `group_size` seats per
   call to `next_batch()`. If you set `group_size = 1`, you get strictly
   sequential prompt->K completions->reward->apply loop behavior.
2) **Feature parity**: Uses the same validation, reward computation, phase
   advancement, and metrics accounting as your previous TurnBatcher.
3) **Stateless w.r.t. Trainer**: All scheduling state (which seat acts next,
   per-phase eval counts, etc.) is internal to the runner; Trainer doesn't change.

How it works
------------
- Each call to `next_batch()` walks the games/players in a deterministic order
  and returns at most `group_size` prompts (1 seat == 1 prompt), along with the
  associated (game, player) metadata.
- `apply_actions()` is called with exactly those metadata entries, and applies
  the completions in the same order, producing rewards and advancing the game(s)
  as needed.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import random
import re
from collections import defaultdict

from .game import Game
from .config import GameConfiguration
from .player import Player
from .roles import create_role_from_name
from .interaction_handler import InteractionHandler
from .grammar import ToSSimGrammarParser, validate_action, get_action_reward
from .prompt_builder import build_training_prompt

class RunnerWithRewards:
    """
    A sequential, reward-returning environment for GRPO.

    Public API matches TurnBatcher:
        next_batch() -> (prompts: List[str], meta: List[tuple[Game, Player]])
        apply_actions(meta, completions) -> List[float]
        get_batch_stats() -> Dict[str, Any]

    Parameters
    ----------
    num_games : int
        Number of concurrent ToSSim games to simulate.
    active_seats_per_game : int
        (Kept for interface parity; we still honor 3-turn-per-phase logic
        internally. You can ignore this or use it to bound per-game sampling.)
    model_name : str
        Passed through to prompt_builder to let you specialize prompts.
    group_size : int
        **Critical**: maximum number of seats you expose to the trainer per call.
        If you set group_size=1, generation is strictly sequential across seats.
        If you set group_size=3, you get micro-batches of size 3, etc.
    evals_per_phase : int
        Number of turns each living player may take per phase before you advance.
        Mirrors the old "3 evals" logic.
    max_days : int
        Hard cutoff to reset stalled / long games.
    """

    def __init__(
        self,
        num_games: int,
        active_seats_per_game: int,
        model_name: str = "default",
        *,
        group_size: int = 1,
        evals_per_phase: int = 3,
        max_days: int = 7,
    ) -> None:
        self.num_games = num_games
        self.active_seats_per_game = active_seats_per_game
        self.model_name = model_name
        self.group_size = max(1, group_size)
        self.evals_per_phase = evals_per_phase
        self.max_days = max_days

        self.games: List[Game] = []
        self._initialize_games()

        # Per-game, per-player eval counters (turns taken in current phase)
        self.eval_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Round-robin cursor for fairness / determinism
        self._cursor: Tuple[int, int] = (0, 0)  # (game_index, player_index)

        # Metrics
        self.metrics = {
            "total_actions": 0,
            "malformed_actions": 0,
            "illegal_actions": 0,
            "legal_actions": 0,
        }

        self.grammar_parser = ToSSimGrammarParser()

    # ------------------------------------------------------------------
    # Public API (used by GRPO.Trainer)
    # ------------------------------------------------------------------
    def next_batch(self) -> Tuple[List[str], List[Tuple[Game, Player]]]:
        """
        Return up to `group_size` prompts + metadata for seats that are ready to act
        (sequentially surfaced). If nothing is ready, we still advance phases / reset
        games as required and return an empty batch.
        """
        print(f"[RunnerWithRewards] next_batch() called")
        prompts: List[str] = []
        meta: List[Tuple[Game, Player]] = []
        
        
        # We will emit at most group_size seats this call.
        remaining = self.group_size

        # Iterate deterministically across games/players, resuming from self._cursor
        gi, pi = self._cursor

        visited_games = 0
        while remaining > 0 and visited_games < len(self.games):
            game = self.games[gi]

            # Auto-reset if needed
            if game.game_is_over() or game.day > self.max_days:
                self._reset_game(gi)
                game = self.games[gi]

            players = [p for p in game.players if p.is_alive]
            if not players:
                # Nothing to do in this game right now, move on
                gi = (gi + 1) % self.num_games
                visited_games += 1
                pi = 0
                continue

            # Make sure player index is in-bounds
            if pi >= len(players):
                pi = 0

            start_pi = pi
            looped_players = 0
            while remaining > 0 and looped_players < len(players):
                player = players[pi]
                if self.eval_counts[gi][player.id] < self.evals_per_phase:
                    # Seat is eligible → add to batch
                    prompt_obj = build_training_prompt(game, player, model_name=self.model_name)
                    prompt = prompt_obj.text   # <- keep returning plain strings from next_batch()
                    prompts.append(prompt)
                    meta.append((game, player))
                    remaining -= 1
                    # Advance cursor to *next* player for the next batch
                    pi = (pi + 1) % len(players)
                    looped_players += 1
                    # If this micro-batch is full, break
                    if remaining == 0:
                        break
                else:
                    # This player has exhausted their evals this phase → skip
                    pi = (pi + 1) % len(players)
                    looped_players += 1

            # Advance to the next game if we didn’t fill the batch yet
            gi = (gi + 1) % self.num_games
            visited_games += 1
            pi = 0

        # Update cursor to where we’ll resume scanning next call
        self._cursor = (gi, pi)

        if not prompts:
            # Nothing was ready → try to advance phases, then return empty batch
            self._advance_games()
        print(f"[env] next_batch -> {len(prompts)} prompts (cursor={self._cursor})")
        return prompts, meta

    def apply_actions(
        self,
        metadata: List[Tuple[Game, Player]],
        completions: List[str],
    ) -> List[float]:
        """
        Apply completions to the simulator, compute rewards, advance phases when
        everyone has completed their quota, and return the reward list.
        """
        rewards: List[float] = []
        for (game, player), completion in zip(metadata, completions):
            # Extract last well-formed think/action block (handles chaining)
            action_text = self._extract_last_action(completion)

            status, error_code, _detail = validate_action(action_text, game, player)
            reward = get_action_reward(status, error_code)
            rewards.append(reward)

            self.metrics["total_actions"] += 1
            final_action = action_text
            if status == "MALFORMED":
                self.metrics["malformed_actions"] += 1
                final_action = (
                    "<think>I think it is better to sit back and weigh my options</think><wait/>"
                )
            elif status == "ILLEGAL":
                self.metrics["illegal_actions"] += 1
                final_action = (
                    "<think>I think it is better to sit back and weigh my options</think><wait/>"
                )
            else:
                self.metrics["legal_actions"] += 1

            # Execute the action (or fallback)
            handler = InteractionHandler(game)
            handler.parse_and_execute(player, final_action)

            # Increment eval count for this player's phase
            gi = self.games.index(game)
            self.eval_counts[gi][player.id] += 1

        # After processing the micro-batch, advance games if all acted
        self._advance_games()
        return rewards

    def get_batch_stats(self) -> Dict[str, Any]:
        total_actions = self.metrics["total_actions"]
        parsing_accuracy = (
            1.0 if total_actions == 0 else (self.metrics["legal_actions"] / total_actions)
        )
        return {
            "total_games": self.num_games,
            "metrics": dict(self.metrics),
            "parsing_accuracy": parsing_accuracy,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _initialize_games(self) -> None:
        for _ in range(self.num_games):
            self.games.append(self._create_new_game())

    def _create_new_game(self) -> Game:
        config = GameConfiguration(game_mode="Ranked Practice")
        # Random but reproducible-ish name pool
        name_pool = [
            "Alice", "Bob", "Charlie", "Daphne", "Eve", "Mallory", "Oscar", "Peggy",
            "Sybil", "Trent", "Victor", "Walter", "Yvonne", "Zane", "Nina", "Liam",
            "Mona", "Igor", "Jane", "Kevin", "Luna", "Mia", "Noah", "Olivia", "Paul",
            "Quinn", "Rita", "Sam", "Tina", "Uma", "Vera", "Wade", "Xena", "Yuri", "Zara"
        ]
        random.shuffle(name_pool)
        player_names = name_pool[: len(config.role_list)]
        players: List[Player] = []
        for i, role_name in enumerate(config.role_list):
            player_name = player_names[i]
            role = create_role_from_name(role_name)
            players.append(Player(player_name, role))
        return Game(config, players)

    def _reset_game(self, game_index: int) -> None:
        print(f"[RunnerWithRewards] Resetting game {game_index} …")
        self.games[game_index] = self._create_new_game()
        if game_index in self.eval_counts:
            del self.eval_counts[game_index]

    def _extract_last_action(self, completion: str) -> str:
        # Grab the last <think>...</think><action/>-ish block
        pattern = r"<think>.*?</think>\s*<[a-zA-Z_]+(?:/>|.*?</[a-zA-Z_]+>)"
        matches = re.findall(pattern, completion, re.DOTALL)
        if not matches:
            return completion
        return matches[-1]

    def _advance_games(self) -> None:
        """
        Check each game; if all living players have exhausted their
        evals_per_phase turns, advance the phase and reset their counters.
        """
        for gi, game in enumerate(self.games):
            alive_players = [p for p in game.players if p.is_alive]
            if not alive_players:
                continue
            all_acted = all(
                self.eval_counts[gi][p.id] >= self.evals_per_phase for p in alive_players
            )
            if all_acted:
                print(
                    f"[RunnerWithRewards] Advancing phase for game {gi} "
                    f"from {getattr(game.phase, 'name', str(game.phase))} …"
                )
                game.advance_phase()
                print(
                    f"[RunnerWithRewards] New phase for game {gi}: "
                    f"{getattr(game.phase, 'name', str(game.phase))}"
                )
                self.eval_counts[gi].clear()
