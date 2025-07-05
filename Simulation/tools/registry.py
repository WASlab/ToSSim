from __future__ import annotations

"""Tool registry and simple execution router.

This module is responsible for:
1. Loading every `*.json` file in the same directory and exposing their
   contents via :pyfunc:`get_tool_registry`.
2. Providing a minimal dispatch layer so the game loop (or test harness)
   can execute a tool call given a tag name and raw argument string.

Only the **few** tools required for current prototypes are implemented.  When
new tool JSON files are added you simply need to extend
:pydata:`_TOOL_EXECUTORS` with a callable that accepts the raw argument string
and returns a *string* (what will be placed inside an `<observation>` tag).
"""

from pathlib import Path
import json
from typing import Callable, Dict, Any

TOOLS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def _load_tool_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _discover_tool_specs() -> Dict[str, dict[str, Any]]:
    specs: Dict[str, dict[str, Any]] = {}
    for json_path in TOOLS_DIR.glob("*.json"):
        try:
            spec = _load_tool_json(json_path)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in tool spec {json_path}: {exc}") from exc

        name = spec.get("name")
        if not name:
            raise ValueError(f"Tool file {json_path} missing mandatory 'name' field")
        specs[name] = spec
    return specs


_TOOL_SPECS = _discover_tool_specs()

# ---------------------------------------------------------------------------
# Concrete executors
# ---------------------------------------------------------------------------


def _exec_get_role(argument: str) -> str:
    """Return public details of a role.

    The implementation instantiates the corresponding Role class (if found)
    and converts it into the lightweight RoleCard representation from
    :pymod:`inference.templates.prompt_builder`.
    """
    from Simulation.enums import RoleName  # imported lazily to avoid circulars
    from Simulation.roles import role_map  # noqa: WPS433
    from inference.templates.prompt_builder import build_role_card  # noqa: WPS433

    role_name_clean = argument.strip()
    try:
        role_enum = RoleName(role_name_clean)
    except ValueError:
        # Try fuzzy match (case-insensitive exact string)
        matches = [e for e in RoleName if e.value.lower() == role_name_clean.lower()]
        if not matches:
            return f"No role named '{role_name_clean}'."
        role_enum = matches[0]

    role_cls = role_map.get(role_enum)
    if role_cls is None:
        return f"Role '{role_name_clean}' not implemented in simulator."

    role_instance = role_cls()
    role_card = build_role_card(role_instance)
    # JSON-encode for a structured observation but keep it human-readable.
    return json.dumps({"role_info": role_card.to_dict()}, indent=2)


def _exec_chat_history(argument: str, *, game=None, player=None) -> str:
    """Return chat history for a specific day or night.
    
    Usage: <chat_history>Day1</chat_history> or <chat_history>Night2</chat_history>
    """
    if not game or not player:
        return "Error: chat history lookup requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    arg_clean = argument.strip().replace(" ", "").replace("_", "").lower()
    
    # Parse the day/night number
    if arg_clean.startswith("day"):
        try:
            day_num = int(arg_clean[3:])
            is_night = False
        except ValueError:
            return "Error: Invalid day format. Use 'Day1', 'Day2', etc."
    elif arg_clean.startswith("night"):
        try:
            night_num = int(arg_clean[5:])
            is_night = True
            # Night N corresponds to game.day = N (not N-1)
            # Night 1 happens during game.day=1, Night 2 during game.day=2, etc.
            day_num = night_num
        except ValueError:
            return "Error: Invalid night format. Use 'Night1', 'Night2', etc."
    else:
        return "Error: Format must be 'Day1', 'Day2', 'Night1', 'Night2', etc."
    
    # Check if the requested period is before current day - 1 (history only)
    if game.day <= 1:
        return "Error: No previous chat history available yet."
    
    if day_num >= game.day:
        return "Error: Cannot view chat history from current or future periods."
    
    # Get chat history from the ChatManager
    return game.chat.get_chat_history(player, day_num, is_night)


def _exec_graveyard(argument: str, *, game=None, player=None) -> str:
    """Return detailed information about a dead player.
    
    Usage: <graveyard>PlayerName</graveyard>
    """
    if not game or not player:
        return "Error: graveyard lookup requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    player_name = argument.strip()
    if not player_name:
        return "Error: You must specify a player name."
    
    # Find the dead player
    dead_player = None
    for p in game.graveyard:
        if p.name.lower() == player_name.lower():
            dead_player = p
            break
    
    if not dead_player:
        return f"Error: No dead player named '{player_name}' found in the graveyard."
    
    # Build detailed death information
    result = f"{dead_player.name} was found dead.\n"
    result += f"They were a {dead_player.role.name.value}.\n"
    
    # Add will information if available
    if hasattr(dead_player, 'was_cleaned') and dead_player.was_cleaned:
        result += "Their will was cleaned and could not be found."
    elif hasattr(dead_player, 'was_forged') and dead_player.was_forged:
        result += "Their will was forged by a member of the Mafia."
    elif hasattr(dead_player, 'last_will_bloodied') and dead_player.last_will_bloodied:
        result += "Their will was too bloody to read."
    elif dead_player.last_will:
        result += f"Will: {dead_player.last_will}"
    else:
        result += "No will was found."
    
    return result


def _exec_check_will(argument: str, *, game=None, player=None) -> str:
    """Return the current player's last will.
    
    Usage: <check_will></check_will> or <check_will>self</check_will>
    """
    if not game or not player:
        return "Error: will check requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    if player.last_will:
        return f"Your current last will:\n{player.last_will}"
    else:
        return "You have not written a last will yet."


def _exec_view_will(argument: str, *, game=None, player=None) -> str:
    """Return a dead player's last will.
    
    Usage: <view_will>DeadPlayerName</view_will>
    """
    if not game or not player:
        return "Error: will viewing requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    player_name = argument.strip()
    if not player_name:
        return "Error: You must specify a player name."
    
    # Find the dead player
    dead_player = None
    for p in game.graveyard:
        if p.name.lower() == player_name.lower():
            dead_player = p
            break
    
    if not dead_player:
        return f"Error: No dead player named '{player_name}' found."
    
    # Check if will is viewable
    if hasattr(dead_player, 'was_cleaned') and dead_player.was_cleaned:
        return f"{dead_player.name}'s will was cleaned and cannot be read."
    elif hasattr(dead_player, 'was_forged') and dead_player.was_forged:
        return f"{dead_player.name}'s will was forged by a member of the Mafia."
    elif hasattr(dead_player, 'last_will_bloodied') and dead_player.last_will_bloodied:
        return f"{dead_player.name}'s will was too bloody to read."
    elif dead_player.last_will:
        return f"{dead_player.name}'s last will:\n{dead_player.last_will}"
    else:
        return f"{dead_player.name} did not leave a will."


# Mapping: tool name -> executor
_TOOL_EXECUTORS: Dict[str, Callable[[str], str]] = {
    "get_role": _exec_get_role,
    "chat_history": _exec_chat_history,
    "graveyard": _exec_graveyard,
    "check_will": _exec_check_will,
    "view_will": _exec_view_will,
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_tool_registry() -> Dict[str, dict[str, Any]]:
    """Return *copy* of the registry dict so callers cannot mutate internals."""
    return {k: v.copy() for k, v in _TOOL_SPECS.items()}


def execute_tool(tool_name: str, raw_argument: str, *, game=None, player=None) -> str:
    """Execute *tool_name* with *raw_argument* and return observation text.

    If the tool is unknown or lacks an executor, a friendly error string is
    returned instead of raising to avoid crashing the simulation.
    
    Parameters:
    • *game* and *player* provide context for tools that need access to game state.
    """
    exec_fn = _TOOL_EXECUTORS.get(tool_name)
    if exec_fn is None:
        return f"Tool '{tool_name}' is not recognised or not yet implemented."
    try:
        # Check if the executor accepts game context
        import inspect
        sig = inspect.signature(exec_fn)
        if 'game' in sig.parameters:
            return exec_fn(raw_argument, game=game, player=player)
        else:
            return exec_fn(raw_argument)
    except Exception as exc:  # noqa: BLE001
        # Never propagate, tool errors should be safe for the agent to read.
        return f"Error executing {tool_name}: {exc}" 