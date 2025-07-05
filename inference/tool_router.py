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
    • *patched_text* is the assistant message with an auto-inserted `</think>`
      (if it was missing) so that tags are balanced.
    • *observation* is the string returned by the tool executor or *None* if no
      tool tag was found.
    
    Parameters:
    • *game* and *player* provide context for tools that need access to game state.
    """
    match = _TOOL_TAG_RE.search(raw_text)
    if match is None:
        return raw_text, None

    tool_name, arg = match.group(1), match.group(2).strip()

    # Ignore tags that are not registered tools (e.g., <speak>, <whisper>, <vote>)
    if tool_name not in _TOOL_NAMES:
        return raw_text, None

    tag_start = match.start()

    # ------------------------------------------------------------------
    # Ensure the tool call lives inside an *open* <think> block
    # ------------------------------------------------------------------
    think_open = raw_text.rfind("<think>", 0, tag_start)
    think_close = raw_text.rfind("</think>", 0, tag_start)

    if think_open == -1 or (think_close != -1 and think_close > think_open):
        # Tool called outside an open <think> block → error observation
        error_obs = f"Error: tool '{tool_name}' must be invoked inside <think>."
        return raw_text, error_obs

    # Auto-close <think> if it is still open at the tool position
    patched_text = raw_text
    if think_close == -1 or think_close < think_open:
        patched_text = (
            raw_text[:tag_start] + "</think>" + raw_text[tag_start:]
        )

    # ------------------------------------------------------------------
    # Execute the tool (unknown tools handled inside execute_tool)
    # ------------------------------------------------------------------
    observation = execute_tool(tool_name, arg, game=game, player=player)
    return patched_text, observation

