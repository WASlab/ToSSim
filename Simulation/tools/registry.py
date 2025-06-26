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


# Mapping: tool name -> executor
_TOOL_EXECUTORS: Dict[str, Callable[[str], str]] = {
    "get_role": _exec_get_role,
    # additional tools go here
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_tool_registry() -> Dict[str, dict[str, Any]]:
    """Return *copy* of the registry dict so callers cannot mutate internals."""
    return {k: v.copy() for k, v in _TOOL_SPECS.items()}


def execute_tool(tool_name: str, raw_argument: str) -> str:
    """Execute *tool_name* with *raw_argument* and return observation text.

    If the tool is unknown or lacks an executor, a friendly error string is
    returned instead of raising to avoid crashing the simulation.
    """
    exec_fn = _TOOL_EXECUTORS.get(tool_name)
    if exec_fn is None:
        return f"Tool '{tool_name}' is not recognised or not yet implemented."
    try:
        return exec_fn(raw_argument)
    except Exception as exc:  # noqa: BLE001
        # Never propagate, tool errors should be safe for the agent to read.
        return f"Error executing {tool_name}: {exc}" 