from __future__ import annotations

"""Light-weight XML tool-call detector and executor.

This helper inspects a completed chunk of assistant output for the **first**
`<tool_name>…</tool_name>` tag, executes the tool via
`Simulation.tools.registry.execute_tool`, and returns *(patched_text,
observation)* so the caller can inject an `<observation>` block and resume the
LLM.

Key rules implemented here:
1. Tool tags **must** appear inside an open `<think>` … `</think>` block.
2. If a tool tag is found while the `<think>` tag is still open (no closing
   `</think>` yet), the router automatically inserts the closing tag *just
   before* the tool call so the transcript remains well-formed.
3. If the tool is unknown, the returned observation is an error string so the
   agent can read what went wrong instead of the simulation crashing.
"""

import re
from typing import Tuple, Optional, TYPE_CHECKING

from Simulation.tools.registry import execute_tool, get_tool_registry

if TYPE_CHECKING:
    from Simulation.game import Game
    from Simulation.player import Player

_TOOL_TAG_RE = re.compile(r"<(?P<name>[a-zA-Z_][\w]*)>(?P<arg>.*?)</\1>", re.DOTALL)
_TOOL_NAMES = set(get_tool_registry().keys())


# Public API -----------------------------------------------------------------

def apply_first_tool_call(raw_text: str, *, game: Optional["Game"] = None, player: Optional["Player"] = None) -> Tuple[str, Optional[str]]:
    """Detect and execute the first tool call in *raw_text*.

    Returns `(patched_text, observation)` where:
    • *patched_text* is the assistant message unchanged (no patching needed).
    • *observation* is the string returned by the tool executor or *None* if no
      tool tag was found.
    
    Parameters:
    • *game* and *player* provide context for tools that need access to game state.
    """
    # Look for any registered tool tag in the text
    for tool_name in _TOOL_NAMES:
        # Create a specific regex for this tool
        tool_regex = re.compile(f"<{re.escape(tool_name)}>(.*?)</{re.escape(tool_name)}>", re.DOTALL)
        match = tool_regex.search(raw_text)
        if match:
            arg = match.group(1).strip()
            # ------------------------------------------------------------------
            # Execute the tool (unknown tools handled inside execute_tool)
            # ------------------------------------------------------------------
            observation = execute_tool(tool_name, arg, game=game, player=player)
            return raw_text, observation
    
    return raw_text, None

