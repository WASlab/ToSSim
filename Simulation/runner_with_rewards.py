# Simulation/runner_with_rewards.py
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

__all__ = ["RunnerWithRewards"]

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
    """Build a prompt but never explode the env if prompt_builder raises."""
    try:
        p = build_training_prompt(game, player, model_name=model_name)
        return p.text
    except Exception as e:
        print(f"[RunnerWithRewards] build_training_prompt failed for {player.name}: {e}")
        return SAFE_FALLBACK_PROMPT

class RunnerWithRewards:
    """
    Canonical sequential GRPO environment with hard progress guarantees.
    """

    def __init__(
        self,
        num_games: int,
        active_seats_per_game: int,
        model_name: str = "default",
        *,
        group_size: int | None = None,
        evals_per_phase: int = 3,
        max_days: int = 7,
        max_empty_ticks: int = 1000,
    ) -> None:
        self.num_games = num_games
        self.active_seats_per_game = active_seats_per_game
        self.model_name = model_name
        self.group_size = max(1, group_size if group_size is not None else active_seats_per_game)
        self.evals_per_phase = evals_per_phase
        self.max_days = max_days

        self.games: List[Game] = []
        self._initialize_games()

        self.eval_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._cursor: Tuple[int, int] = (0, 0)

        self.metrics = {
            "total_actions": 0,
            "malformed_actions": 0,
            "illegal_actions": 0,
            "legal_actions": 0,
            "empty_batches": 0,
        }

        self.grammar_parser = ToSSimGrammarParser()
        self._consecutive_empty_batches = 0
        self._max_empty_ticks = max_empty_ticks

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def next_batch(self) -> Tuple[List[str], List[Tuple[Game, Player]]]:
        attempts = 0
        max_attempts = max(1, self.num_games)

        while attempts < max_attempts:
            print(f"[RunnerWithRewards] next_batch() attempt {attempts + 1}/{max_attempts}")
            prompts: List[str] = []
            meta: List[Tuple[Game, Player]] = []

            remaining = self.group_size
            gi, pi = self._cursor
            visited_games = 0

            while remaining > 0 and visited_games < len(self.games):
                game = self.games[gi]

                if game.game_is_over() or getattr(game, "day", 1) > self.max_days:
                    self._reset_game(gi)
                    game = self.games[gi]

                players = [p for p in game.players if p.is_alive]
                if not players:
                    gi = (gi + 1) % self.num_games
                    visited_games += 1
                    pi = 0
                    continue

                if pi >= len(players):
                    pi = 0

                looped_players = 0
                while remaining > 0 and looped_players < len(players):
                    player = players[pi]
                    if self.eval_counts[gi][player.id] < self.evals_per_phase:
                        prompt = _safe_build_prompt(game, player, self.model_name)
                        prompts.append(prompt)
                        meta.append((game, player))
                        remaining -= 1
                        pi = (pi + 1) % len(players)
                        looped_players += 1
                        if remaining == 0:
                            break
                    else:
                        pi = (pi + 1) % len(players)
                        looped_players += 1

                gi = (gi + 1) % self.num_games
                visited_games += 1
                pi = 0

            self._cursor = (gi, pi)

            if prompts:
                self._consecutive_empty_batches = 0
                print(f"[env] next_batch -> {len(prompts)} prompts (cursor={self._cursor})")
                return prompts, meta

            # No prompts
            attempts += 1
            self._consecutive_empty_batches += 1
            self.metrics["empty_batches"] += 1
            self._advance_games(force= self._consecutive_empty_batches >= self._max_empty_ticks)

        print(f"[env] next_batch -> 0 prompts after {max_attempts} attempts (cursor={self._cursor})")
        return [], []

    def apply_actions(
        self,
        metadata: List[Tuple[Game, Player]],
        completions: List[str],
    ) -> List[float]:
        rewards: List[float] = []
        for (game, player), completion in zip(metadata, completions):
            action_text = self._extract_last_action(completion)

            status, error_code, _detail = validate_action(action_text, game, player)
            reward = get_action_reward(status, error_code)
            rewards.append(reward)

            self.metrics["total_actions"] += 1
            final_action = action_text
            if status in ("MALFORMED", "ILLEGAL"):
                key = "malformed_actions" if status == "MALFORMED" else "illegal_actions"
                self.metrics[key] += 1
                final_action = (
                    "<think>I think it is better to sit back and weigh my options</think><wait/>"
                )
            else:
                self.metrics["legal_actions"] += 1

            handler = InteractionHandler(game)
            try:
                handler.parse_and_execute(player, final_action)
            except Exception as e:
                print(f"[RunnerWithRewards] parse_and_execute crashed: {e}")

            gi = self.games.index(game)
            self.eval_counts[gi][player.id] += 1

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

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _initialize_games(self) -> None:
        for _ in range(self.num_games):
            self.games.append(self._create_new_game())

    def _create_new_game(self) -> Game:
        config = GameConfiguration(game_mode="Ranked Practice")
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
        pattern = r"<think>.*?</think>\s*<[a-zA-Z_]+(?:/>|.*?</[a-zA-Z_]+>)"
        matches = re.findall(pattern, completion, re.DOTALL)
        if not matches:
            return completion
        return matches[-1]

    def _advance_games(self, force: bool = False) -> None:
        """
        Advance phases when all alive players exhausted evals_per_phase.
        If `force=True`, advance regardless (to break pathological stalls).
        """
        for gi, game in enumerate(self.games):
            alive_players = [p for p in game.players if p.is_alive]
            if not alive_players:
                continue
            all_acted = all(
                self.eval_counts[gi][p.id] >= self.evals_per_phase for p in alive_players
            )
            if all_acted or force:
                print(
                    f"[RunnerWithRewards] Advancing phase for game {gi} "
                    f"(force={force}) from {getattr(game.phase, 'name', str(game.phase))} …"
                )
                try:
                    game.advance_phase()
                except Exception as e:
                    print(f"[RunnerWithRewards] game.advance_phase crashed: {e}")
                print(
                    f"[RunnerWithRewards] New phase for game {gi}: "
                    f"{getattr(game.phase, 'name', str(game.phase))}"
                )
                self.eval_counts[gi].clear()
