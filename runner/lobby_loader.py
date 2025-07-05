from __future__ import annotations

"""runner.lobby_loader – parse & validate lobby YAML files.

This helper provides one public function:

    load_lobby(path: str | None) -> "LobbyConfig"

If *path* is *None* the function falls back to ``configs/lobby_default.yaml``.
The loader performs minimal but strict validation so that ``MatchRunner`` can
rely on the returned structure being complete and coherent.

Accepted schema (YAML):

```
# Optional – overrides GameConfiguration defaults
game:
  mode: "Ranked Practice"
  coven: false

agents:                # exactly 15 entries
  - id: Player1        # required
    model: ToSSim/misaligned-gemma-3-27b-QDoRA-8bit   # required
    quantization: 8bit                                # required (4bit|8bit|bf16)
    misaligned: true                                  # required (bool) – if missing, inferred
    personality: default                              # optional
```
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

import yaml

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentSpec:
    id: str
    model: str
    quantization: str
    misaligned: bool
    personality: Optional[str] = None
    # These are filled later by the Game environment – kept here for logging
    role: Optional[str] = None
    faction: Optional[str] = None


@dataclass
class GameSpec:
    mode: str = "Ranked Practice"
    coven: bool = False


@dataclass
class LobbyConfig:
    game: GameSpec
    agents: List[AgentSpec]

    # Convenience helper: mapping id → model (what MatchRunner expects)
    def model_map(self) -> Dict[str, str]:
        return {a.id: a.model for a in self.agents}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


_ALLOWED_QUANT = {"4bit", "8bit", "bf16"}
_DEFAULT_LOBBY_PATH = Path("configs/lobby_default.yaml")


def _infer_misaligned(model_name: str) -> bool:
    return "misaligned" in model_name.lower()


def _infer_quant(model_name: str) -> Optional[str]:
    name_l = model_name.lower()
    if "4bit" in name_l or "fp4" in name_l:
        return "4bit"
    if "8bit" in name_l or "fp8" in name_l:
        return "8bit"
    if "bf16" in name_l or "fp16" in name_l or "16bit" in name_l:
        return "bf16"
    return None


def _validate_agent(raw: Dict[str, Any], idx: int) -> AgentSpec:
    def err(msg: str):
        raise ValueError(f"Agent[{idx}] {msg}")

    unknown_keys = set(raw) - {
        "id",
        "model",
        "quantization",
        "misaligned",
        "personality",
        "role",
        "faction",
    }
    if unknown_keys:
        warnings.warn(
            f"Agent[{idx}] unknown keys {sorted(unknown_keys)} – ignored.",
            RuntimeWarning,
        )

    # Required ---------------------------------------------------------
    aid = raw.get("id")
    if not isinstance(aid, str) or not aid:
        err("'id' must be a non-empty string")

    model = raw.get("model")
    if not isinstance(model, str) or not model:
        err("'model' must be a non-empty string (HF model id)")

    # Quantisation ------------------------------------------------------
    quant = raw.get("quantization")
    if quant is None:
        quant = _infer_quant(model)
    if quant not in _ALLOWED_QUANT:
        err(
            "'quantization' must be one of {" + ", ".join(sorted(_ALLOWED_QUANT)) + "}"\
            + " (or inferable from model name)"
        )

    # Misaligned flag ---------------------------------------------------
    misaligned = raw.get("misaligned")
    if misaligned is None:
        misaligned = _infer_misaligned(model)
    if not isinstance(misaligned, bool):
        err("'misaligned' must be boolean (true/false)")

    personality = raw.get("personality")
    if personality is not None and not isinstance(personality, str):
        err("'personality' must be a string if provided")

    return AgentSpec(
        id=aid,
        model=model,
        quantization=quant,
        misaligned=misaligned,
        personality=personality,
        role=raw.get("role"),
        faction=raw.get("faction"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_lobby(path: str | Path | None = None) -> LobbyConfig:
    """Load & validate a lobby YAML file.

    Parameters
    ----------
    path : str | pathlib.Path | None
        Path to YAML file.  If *None*, falls back to ``configs/lobby_default.yaml``.
    """

    yaml_path = Path(path) if path else _DEFAULT_LOBBY_PATH
    if not yaml_path.exists():
        raise FileNotFoundError(f"Lobby config not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML syntax in {yaml_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping (dict)")

    unknown_top = set(raw) - {"game", "agents", "narrator"}
    if unknown_top:
        warnings.warn(
            f"Unknown top-level keys {sorted(unknown_top)} – ignored.", RuntimeWarning
        )

    # ------------------------------------------------------------------
    # Game block
    # ------------------------------------------------------------------
    game_raw = raw.get("game", {}) or {}
    if not isinstance(game_raw, dict):
        raise ValueError("'game' section must be a mapping")

    mode = game_raw.get("mode", "Ranked Practice")
    coven = game_raw.get("coven", False)
    if not isinstance(mode, str):
        raise ValueError("game.mode must be string")
    if not isinstance(coven, bool):
        raise ValueError("game.coven must be boolean")

    game_spec = GameSpec(mode=mode, coven=coven)

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    agents_raw = raw.get("agents")
    if not isinstance(agents_raw, list):
        raise ValueError("'agents' must be a list of mappings")
    if len(agents_raw) != 15:
        raise ValueError("Exactly 15 agents are required (got %d)" % len(agents_raw))

    agents: List[AgentSpec] = []
    seen_ids: set[str] = set()
    for idx, a_raw in enumerate(agents_raw):
        if not isinstance(a_raw, dict):
            raise ValueError(f"Agent[{idx}] must be a mapping")
        spec = _validate_agent(a_raw, idx)
        if spec.id in seen_ids:
            raise ValueError(f"Duplicate agent id '{spec.id}' in agents list")
        seen_ids.add(spec.id)
        agents.append(spec)

    # narrator currently ignored if present – warn if not null
    if raw.get("narrator") not in (None, "null"):
        warnings.warn("'narrator' is reserved for future use and is ignored", RuntimeWarning)

    return LobbyConfig(game=game_spec, agents=agents) 