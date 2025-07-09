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


def _exec_get_time(argument: str, *, game=None, player=None) -> str:
    """Return remaining tokens in the current phase.
    
    Usage: <get_time></get_time>
    """
    if not game or not player:
        return "Error: time lookup requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    # TODO: Implement actual token tracking
    # For now, return a placeholder value
    remaining_tokens = 250  # Placeholder
    phase_name = f"{game.time.name} {game.phase.name}" if hasattr(game, 'phase') else game.time.name
    
    return f"{remaining_tokens} tokens remaining in {phase_name} phase."


def _exec_notebook(argument: str, *, game=None, player=None) -> str:
    """Write to the player's private notebook.
    
    Usage: <notebook>Note content here</notebook>
    """
    if not game or not player:
        return "Error: notebook requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    content = argument.strip()
    if not content:
        return "Error: Cannot write empty note to notebook."
    
    # Initialize notebook if it doesn't exist
    if not hasattr(player, 'notebook'):
        player.notebook = ""
        player.notebook_tokens = 0
    
    # Estimate tokens (rough approximation: 1 token = 4 characters)
    new_tokens = len(content) // 4 + 1
    
    # Check if adding would exceed limit
    NOTEBOOK_LIMIT = 1500
    if player.notebook_tokens + new_tokens > NOTEBOOK_LIMIT:
        # FIFO sliding window - remove content from the beginning
        while player.notebook_tokens + new_tokens > NOTEBOOK_LIMIT and player.notebook:
            # Find first line and remove it
            first_newline = player.notebook.find('\n')
            if first_newline == -1:
                # Only one line left
                removed_tokens = len(player.notebook) // 4 + 1
                player.notebook = ""
                player.notebook_tokens = 0
                break
            else:
                removed_line = player.notebook[:first_newline + 1]
                removed_tokens = len(removed_line) // 4 + 1
                player.notebook = player.notebook[first_newline + 1:]
                player.notebook_tokens -= removed_tokens
    
    # Add new content
    if player.notebook:
        player.notebook += "\n" + content
    else:
        player.notebook = content
    player.notebook_tokens += new_tokens
    
    # Warning message at 85% full
    warning = ""
    if player.notebook_tokens >= int(NOTEBOOK_LIMIT * 0.85):
        warning = "\nWarning: Notebook is 85% full. Consider summarizing old notes as they will be forgotten gradually when the limit is reached."
    
    return f"Note added to notebook. Tokens used: {player.notebook_tokens}/{NOTEBOOK_LIMIT}{warning}"


def _exec_view_notebook(argument: str, *, game=None, player=None) -> str:
    """View the player's private notebook.
    
    Usage: <view_notebook></view_notebook>
    """
    if not game or not player:
        return "Error: notebook viewing requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    # Initialize notebook if it doesn't exist
    if not hasattr(player, 'notebook'):
        player.notebook = ""
        player.notebook_tokens = 0
    
    NOTEBOOK_LIMIT = 1500
    
    if not player.notebook:
        return f"Notebook (0/{NOTEBOOK_LIMIT} tokens):\n\n(Empty)"
    
    return f"Notebook ({player.notebook_tokens}/{NOTEBOOK_LIMIT} tokens):\n\n{player.notebook}"


def _exec_get_executable_actions(argument: str, *, game=None, player=None) -> str:
    """Return valid terminal actions for the current phase.
    
    Usage: <get_executable_actions></get_executable_actions>
    """
    if not game or not player:
        return "Error: action lookup requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    from Simulation.enums import Time, Phase
    
    # Day actions
    if game.time == Time.DAY:
        day_actions = ["speak", "wait"]
        
        # Add voting actions if not in defense/judgement
        if game.phase != Phase.DEFENSE and game.phase != Phase.JUDGEMENT:
            day_actions.extend(["vote", "nominate", "whisper"])
        
        # Role-specific day actions
        if player.role.name.value == "Mayor" and not getattr(player.role, 'revealed', False):
            day_actions.append("reveal")
        if player.role.name.value == "Jailor":
            day_actions.append("jail")
        
        phase_name = f"Day {game.phase.name.title()}"
        return f"Valid actions for {phase_name}: {', '.join(day_actions)}"
    
    # Night actions
    elif game.time == Time.NIGHT:
        night_actions = ["skip", "pass"]
        
        # Role-specific night actions
        role_name = player.role.name.value
        if role_name in ["Sheriff", "Investigator", "Consigliere"]:
            night_actions.append("investigate")
        if role_name in ["Doctor", "Bodyguard", "Crusader", "Guardian Angel"]:
            night_actions.append("protect")
        if role_name == "Vigilante":
            night_actions.append("shoot")
        if role_name in ["Godfather", "Mafioso", "Serial Killer", "Arsonist", "Werewolf"]:
            night_actions.append("kill")
        if role_name == "Jailor" and getattr(player.role, 'jailed_target', None):
            night_actions.append("execute")
        if role_name in ["Escort", "Consort", "Tavern Keeper", "Bootlegger"]:
            night_actions.append("distract")
        if role_name in ["Retributionist", "Necromancer"]:
            night_actions.append("raise")
        if role_name == "Transporter":
            night_actions.append("transport")
        if role_name == "Veteran":
            night_actions.append("alert")
        if role_name == "Tracker":
            night_actions.append("track")
        if role_name == "Lookout":
            night_actions.append("watch")
        if role_name == "Spy":
            night_actions.append("bug")
        if role_name == "Psychic":
            night_actions.append("vision")
        if role_name == "Hex Master":
            night_actions.append("hex")
        if role_name == "Poisoner":
            night_actions.append("poison")
        if role_name == "Medusa":
            night_actions.append("stone")
        if role_name == "Pirate":
            night_actions.append("plunder")
        
        return f"Valid actions for Night (as {role_name}): {', '.join(night_actions)}"
    
    return "Error: Unknown game phase."


def _exec_action_history(argument: str, *, game=None, player=None) -> str:
    """Return the player's action history with results.
    
    Usage: <action_history></action_history>
    """
    if not game or not player:
        return "Error: action history requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    # Initialize action history if it doesn't exist
    if not hasattr(player, 'action_history'):
        player.action_history = []
    
    if not player.action_history:
        return "No action history available yet."
    
    history_text = "Action History:\n"
    for entry in player.action_history:
        phase = entry.get('phase', 'Unknown')
        action = entry.get('action', 'Unknown')
        target = entry.get('target', 'None')
        result = entry.get('result', 'No result recorded')
        
        if target and target != 'None':
            history_text += f"{phase}: {action} {target} - {result}\n"
        else:
            history_text += f"{phase}: {action} - {result}\n"
    
    return history_text.strip()

    return json.dumps({
        "tool_name": "get_role",
        "class": "environment_static",
        "observation": {"role_info": role_card.to_dict()}
    }, indent=2)
# ---------------------------------------------------------------------------
def _exec_get_role_details(argument: str) -> str:
    """Return detailed information about a role.

    This is a placeholder for future expansion, currently returns a simple
    message indicating that the tool is not yet implemented.
    """
    from Simulation.enums import RoleName  # imported lazily to avoid circulars
    from Simulation.roles import role_map  # noqa: WPS433


    json_path = Path(__file__).parent.parent / "reference_data" / "role_details.json"
    
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return f"Role details file not found at {json_path}"
    except json.JSONDecodeError as e:
        return f"Error parsing role details JSON: {e}"


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

    role_details = data.get(role_name_clean, {})

    if not role_details:
            return f"No details found for role '{role_name_clean}'."



def _exec_attributes(argument: str) -> str:
    """Return information about game attributes.
    
    If no argument is provided, returns all attributes.
    If a specific attribute is provided, returns only that attribute.
    If an invalid attribute is provided, returns an error message.
    """    
    # Load attributes data
    attributes_path = Path(__file__).parent.parent / "reference_data" / "attributes.json"
    
    try:
        with attributes_path.open("r", encoding="utf-8") as fh:
            attributes_data = json.load(fh)
    except FileNotFoundError:
        return f"Attributes data file not found at {attributes_path}"
    except json.JSONDecodeError as e:
        return f"Error parsing attributes JSON: {e}"
    
    attribute_name = argument.strip()
    
    # If no specific attribute requested, return all attributes
    if not attribute_name:
        return json.dumps({
            "tool_name": "attributes",
            "class": "environment_static", 
            "observation": {"attribute_info": attributes_data}
        }, indent=2)
    
    # Look for the specific attribute (case-sensitive first, then case-insensitive)
    if attribute_name in attributes_data:
        attribute_info = {attribute_name: attributes_data[attribute_name]}
    else:
        # Try case-insensitive match
        matches = [k for k in attributes_data.keys() if k.lower() == attribute_name.lower()]
        if matches:
            matched_key = matches[0]
            attribute_info = {matched_key: attributes_data[matched_key]}
        else:
            return f"No attribute with the name '{attribute_name}' found."
    
    return json.dumps({
        "tool_name": "attributes",
        "class": "environment_static",
        "observation": {"attribute_info": attribute_info}
    }, indent=2)


def _exec_write_will(argument: str, *, game=None, player=None) -> str:
    """Write or update the player's last will.
    
    Usage: <write_will>My will content here</write_will>
    """
    if not game or not player:
        return "Error: will writing requires game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot write a will while dead."
    
    will_content = argument.strip()
    if not will_content:
        return "Error: Cannot write an empty will."
    
    # Update the player's last will
    player.last_will = will_content
    return "Your last will has been updated."


def _exec_investigation_results(argument: str, *, game=None, player=None) -> str:
    """Return the investigator results chart.
    
    Usage: <investigation_results></investigation_results>
    """
    if not game or not player:
        return "Error: investigation results require game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    # Check if Coven expansion is enabled for different results
    is_coven = getattr(game.config, 'is_coven', False)
    
    if is_coven:
        # Coven expansion investigation results
        results = """Investigator Results (Coven Expansion):
- Investigator, Consigliere, Mayor, Tracker, Plaguebearer
- Lookout, Forger, Witch, Coven Leader
- Sheriff, Executioner, Werewolf, Poisoner
- Framer, Vampire, Jester, Hex Master
- Medium, Janitor, Retributionist, Necromancer, Trapper
- Survivor, Vampire Hunter, Amnesiac, Medusa
- Spy, Blackmailer, Jailor, Guardian Angel
- Escort, Transporter, Consort, Hypnotist
- Doctor, Disguiser, Serial Killer, Potion Master
- Bodyguard, Godfather, Arsonist, Crusader
- Vigilante, Veteran, Mafioso, Pirate, Ambusher
- (No one else)"""
    else:
        # Classic investigation results
        results = """Investigator Results (Classic):
- Investigator, Consigliere, Mayor, Tracker
- Lookout, Forger, Witch
- Sheriff, Executioner, Werewolf
- Framer, Vampire, Jester
- Medium, Janitor, Retributionist
- Survivor, Vampire Hunter, Amnesiac
- Spy, Blackmailer, Jailor
- Escort, Transporter, Consort
- Doctor, Disguiser, Serial Killer
- Bodyguard, Godfather, Arsonist
- Vigilante, Veteran, Mafioso
- (No one else)"""
    
    return results


def _exec_evil_investigation_results(argument: str, *, game=None, player=None) -> str:
    """Return information about consigliere and spy results.
    
    Usage: <evil_investigation_results></evil_investigation_results>
    """
    if not game or not player:
        return "Error: evil investigation results require game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    results = """Evil Investigation Results:

CONSIGLIERE:
- Sees exact role of target
- Coven members with Necronomicon are detection immune
- Hexed players show as "Hex Master"
- Doused players show as "Arsonist"

SPY INTELLIGENCE:
- Sees all Mafia visits: "A member of the mafia visited X last night"
- Sees all Coven visits: "A member of the coven visited X last night"
- Bug results show detailed target information

SPECIAL COVEN MECHANICS:
- Witch, Coven Leader, Potion Master see investigation results of their first target
- Multiple Witches: random one succeeds if targeting same player
- All Coven members gain detection immunity with Necronomicon

INVESTIGATION GROUPS (for Coven roles):
- Witch, Coven Leader, Hex Master, Poisoner all show same investigator result"""
    
    return results


def _exec_victory_conditions(argument: str, *, game=None, player=None) -> str:
    """Return victory condition information for all factions.
    
    Usage: <victory_conditions></victory_conditions>
    """
    if not game or not player:
        return "Error: victory conditions require game context."
    
    # Restrict to living players only
    if not player.is_alive:
        return "Error: You cannot use tools while dead."
    
    results = """Victory Conditions:

TOWN:
- Win: Lynch every criminal and evildoer
- Wins with: Survivors
- Must eliminate: All Mafia, Coven, and Neutral Killing roles

MAFIA:
- Win: Equal or outnumber Town with no hostile neutrals alive
- Wins with: Survivors, Witches (Classic), Executioners/Jesters (if they win)
- Must eliminate: Town and Neutral Killing roles

COVEN:
- Win: Equal or outnumber Town with no hostile neutrals alive  
- Wins with: Survivors, Executioners/Jesters (if they win)
- Must eliminate: Town, Mafia, and Neutral Killing roles

NEUTRAL KILLING:
- Serial Killer: Survive to end, kill everyone else
- Arsonist: Survive to end, kill everyone else
- Werewolf: Survive to end, kill everyone else
- Juggernaut: Survive to end, kill everyone else
- Pestilence: Kill everyone (no allies)

NEUTRAL EVIL:
- Executioner: Get your target lynched, then win with any faction
- Jester: Get lynched by the Town
- Witch (Classic): Win with Mafia

NEUTRAL BENIGN:
- Survivor: Survive to end, wins with any faction
- Amnesiac: Remember a role, then follow that role's win condition
- Guardian Angel: Keep your target alive until end, then win with any faction

SPECIAL NOTES:
- Ties go to Town if only Town/Mafia/Coven remain
- Survivors always win if alive at game end
- Some roles change win conditions (GA→Survivor, Exe→Jester, VH→Vigilante)"""
    
    return results


# Mapping: tool name -> executor
_TOOL_EXECUTORS: Dict[str, Callable[[str], str]] = {
    "get_role": _exec_get_role,
    "get_role_details": _exec_get_role_details,  
    "attributes": _exec_attributes,
    
    "chat_history": _exec_chat_history,
    "graveyard": _exec_graveyard,
    "check_will": _exec_check_will,
    "view_will": _exec_view_will,
    "get_time": _exec_get_time,
    "notebook": _exec_notebook,
    "view_notebook": _exec_view_notebook,
    "get_executable_actions": _exec_get_executable_actions,
    "action_history": _exec_action_history,
    "write_will": _exec_write_will,
    "investigation_results": _exec_investigation_results,
    "evil_investigation_results": _exec_evil_investigation_results,
    "victory_conditions": _exec_victory_conditions,
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