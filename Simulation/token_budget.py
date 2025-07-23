from __future__ import annotations

"""Simulation.token_budget – centralised token budget tracking.

Usage:
    tb = TokenBudgetManager.from_yaml("configs/environment_limits.yaml")
    tb.start_phase("discussion", living=15)
    exhausted = tb.consume("public", 23)  # returns True if phase budget used.

The class keeps *per-phase* aggregate counters plus convenience helpers for
voice channels at night.
"""

from pathlib import Path
from typing import Dict, Optional
import yaml

__all__ = ["TokenBudgetManager"]


class TokenBudgetManager:
    PHASES_WITH_PER_AGENT = {"discussion", "nomination"}

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._phase_budget: int = 0
        self._remaining: Dict[str, int] = {}
        self._current_phase: Optional[str] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cls(cfg)

    # ------------------------------------------------------------------
    # Phase control
    # ------------------------------------------------------------------
    def start_phase(self, phase: str, *, living: int = 0):
        self._current_phase = phase

        if phase in self.PHASES_WITH_PER_AGENT:
            per = self.cfg["per_agent"].get(phase, 0)
            self._phase_budget = per * max(1, living)
        elif phase == "defense":
            self._phase_budget = self.cfg.get("defense", 0)
        elif phase == "judgement":
            self._phase_budget = self.cfg["judgement"].get("total", 0)
        elif phase == "last_words_defendant":
            self._phase_budget = self.cfg["last_words"].get("defendant", 0)
        elif phase == "post_last_words":
            self._phase_budget = self.cfg["last_words"].get("post_reveal_global", 0)
        elif phase == "night_channel":
            self._phase_budget = self.cfg.get("night_channel", 0)
        elif phase == "pre_night":
            self._phase_budget = self.cfg.get("pre_night", 0)
        else:
            self._phase_budget = 0

        self._remaining = {"public": self._phase_budget}

    def set_channel(self, name: str):
        # night channels: allocate separate bucket but same max
        self._remaining[name] = self._phase_budget

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------
    def consume(self, channel: str, tokens: int) -> bool:
        if channel not in self._remaining:
            # default to phase bucket
            self._remaining[channel] = self._phase_budget
        self._remaining[channel] = max(0, self._remaining[channel] - tokens)
        return self._remaining[channel] == 0

    def phase_exhausted(self) -> bool:
        # exhausted when ALL buckets hit zero
        return all(v == 0 for v in self._remaining.values())

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def remaining(self, channel: str = "public") -> int:
        return self._remaining.get(channel, 0)

    def tokens_remaining(self) -> int | None:
        """Return the number of tokens remaining for the current phase, or None if unlimited."""
        if hasattr(self, '_phase_budget'):
            return self._phase_budget
        return None 