"""
Comprehensive prompt building system for ToSSim agents.

This module provides model-specific prompt building with:
1. Static system prompts (role cards, tools, interactions, phase rules)
2. Dynamic user prompts (phase context, roster, graveyard, observations)
3. Model-specific templating (Gemma vs others)
4. Tool categorization (environment_static, environment_dynamic, environment_terminal)
5. Phase-specific UI and voting information

All consumers (training, inference, self-play) should use this module
to ensure consistent prompt formatting across the codebase.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field
import json

from .enums import Time, Phase, RoleName
from .roles import create_role_from_name

if TYPE_CHECKING:
    from .game import Game
    from .player import Player
    from .roles import Role


# ---------------------------------------------------------------------------
# Model Configuration System
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for model-specific prompt formatting."""
    name: str
    has_system_prompt: bool = True
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "</s>"
    
    def format_messages(self, system_prompt: str, user_prompt: str) -> str:
        """Format system and user prompts according to model requirements."""
        if self.has_system_prompt:
            return f"{self.system_token}\n{system_prompt}\n\n{self.user_token}\n{user_prompt}\n\n{self.assistant_token}\n"
        else:
            # Models like Gemma treat system prompt as additional user prompt
            return f"{self.user_token}\n{system_prompt}\n\n{self.user_token}\n{user_prompt}\n\n{self.assistant_token}\n"


# Default model configurations
MODEL_CONFIGS = {
    "gemma": ModelConfig(
        name="gemma",
        has_system_prompt=False,
        system_token="<start_of_turn>user",
        user_token="<start_of_turn>user", 
        assistant_token="<start_of_turn>model",
        end_token="<end_of_turn>"
    ),
    "default": ModelConfig(
        name="default",
        has_system_prompt=True
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration, falling back to default if not found."""
    # Check for exact match first
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Check for partial matches (e.g., "gemma-7b" -> "gemma")
    model_lower = model_name.lower()
    for config_name, config in MODEL_CONFIGS.items():
        if config_name in model_lower:
            return config
    
    return MODEL_CONFIGS["default"]


# ---------------------------------------------------------------------------
# Tool and Interaction System
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """Structured tool specification for prompt building."""
    name: str
    syntax: str
    description: str
    example_input: str
    example_output: str
    tool_class: str = "environment_static"  # environment_static, environment_dynamic, environment_terminal


@dataclass
class InteractionSpec:
    """Structured interaction specification for prompt building."""
    name: str
    syntax: str
    description: str
    example_input: str
    example_output: str
    phases_allowed: List[str] = field(default_factory=list)


def discover_tools() -> List[ToolSpec]:
    """Discover and parse all available tools from the tools directory."""
    tools = []
    tools_dir = Path(__file__).parent / "tools"
    
    # Import the tool registry to get actual implementations
    try:
        from .tools.registry import get_tool_registry
        registry = get_tool_registry()
    except ImportError:
        registry = {}
    
    # Load JSON specifications
    for json_file in tools_dir.glob("*.json"):
        if json_file.name == "__init__.py":
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                spec = json.load(f)
                
            tool_name = spec.get("name", json_file.stem)
            tool_class = spec.get("class", "environment_static")
            description = spec.get("description", "")
            
            # Generate syntax based on tool name
            if tool_name in ["view_will", "check_will", "view_notebook"]:
                syntax = f"</{tool_name}>"
            elif tool_name == "notebook":
                syntax = f"<{tool_name}>content</{tool_name}>"
            elif tool_name in ["whisper"]:
                syntax = f'<{tool_name} target="PlayerName">message</{tool_name}>'
            else:
                syntax = f"<{tool_name}>argument</{tool_name}>"
            
            # Extract examples if available
            examples = spec.get("examples", [])
            example_input = f"<{tool_name}>example</{tool_name}>" if not examples else str(examples[0].get("input", ""))
            example_output = "[Tool result]" if not examples else str(examples[0].get("output", ""))
            
            # Format examples better
            if tool_name == "view_will":
                example_input = "</view_will>"
                example_output = '[Your current will: "Investigated Player5 on Night 2, result was Town."]'
            elif tool_name == "role_details" or tool_name == "get_role_details":
                example_input = "<role_details>William</role_details>"
                example_output = "[Player William's role is Bodyguard.]"
            elif tool_name == "write_will":
                example_input = "<write_will>I healed Player3 on Night 2.</write_will>"
                example_output = "[Your will has been updated.]"
            elif tool_name == "chat_history":
                example_input = "<chat_history>Day1</chat_history>"
                example_output = "[Day 1 chat history: ...]"
            elif tool_name == "graveyard":
                example_input = "<graveyard>John</graveyard>"
                example_output = "[John (Mafioso) — died Day 1. Last Will: 'I'm not Mafia.']"
            elif tool_name == "notebook":
                example_input = "<notebook>Player3 seems suspicious</notebook>"
                example_output = "[Note added to notebook. Tokens used: 45/1500]"
            
            tools.append(ToolSpec(
                name=tool_name,
                syntax=syntax,
                description=description,
                example_input=example_input,
                example_output=example_output,
                tool_class=tool_class
            ))
        except Exception as e:
            print(f"Warning: Could not parse tool {json_file}: {e}")
    
    return tools


def get_available_interactions() -> List[InteractionSpec]:
    """Get the standard interaction types available to agents."""
    return [
        InteractionSpec(
            name="speak",
            syntax="<speak>message</speak>",
            description="Sends the specified message to the public voice or chat channel. Replace 'message' with the content you wish to say. This message is visible to all players in the game.",
            example_input="<speak>I am not the Mafia, please do not vote for me.</speak>",
            example_output="[Your message has been sent to the public channel.]",
            phases_allowed=["Day Discussion", "Nomination", "Defense (if on trial only)", "Last Words (if being executed only)"]
        ),
        InteractionSpec(
            name="whisper",
            syntax='<whisper target="PlayerName">message</whisper>',
            description="Sends a private message to a specific player. Replace 'PlayerName' with the exact name of the recipient and 'message' with your intended content. Only the target player will receive this message. This is the only tag that uses an attribute for targeting.",
            example_input='<whisper target="Sarah">Trust me, I am on your side.</whisper>',
            example_output="[Your message has been privately delivered to Sarah.]",
            phases_allowed=["Day Discussion", "Nomination", "Defense", "Judgement", "some Night phases (role-dependent)"]
        ),
        InteractionSpec(
            name="vote",
            syntax="<vote>PlayerName</vote>",
            description="During nomination phase, nominates the specified player for trial. During judgement phase, use <vote>guilty</vote>, <vote>innocent</vote>, or <vote>abstain</vote>.",
            example_input="<vote>John</vote> or <vote>guilty</vote>",
            example_output="[Your vote has been recorded.]",
            phases_allowed=["Nomination", "Judgement"]
        ),
        InteractionSpec(
            name="wait",
            syntax="<wait></wait> or </wait>",
            description="Ends your current output without taking any action or sending a message. Can be used to indicate you are waiting or passing on this turn. Both <wait></wait> and </wait> are valid; use either form as appropriate.",
            example_input="</wait>",
            example_output="[No action taken; you are waiting.]",
            phases_allowed=["All phases"]
        )
    ]


# ---------------------------------------------------------------------------
# Role Card System
# ---------------------------------------------------------------------------

@dataclass
class RoleCard:
    """Complete role card information for prompt building."""
    name: str
    faction: str
    alignment: str
    win_condition: str
    passive_abilities: List[str] = field(default_factory=list)
    active_abilities: List[str] = field(default_factory=list)
    attack: str = "None"
    defense: str = "None"
    visit_type: str = "Non-harmful"
    immunities: List[str] = field(default_factory=list)
    unique: bool = False


def build_role_card(role: 'Role') -> RoleCard:
    """Build a comprehensive role card from a role instance."""
    
    # Get basic info
    info = role.get_info() if hasattr(role, "get_info") else {}
    role_name = info.get("name", str(role.name.value) if hasattr(role, 'name') else "Unknown")
    
    # Default mappings for win conditions
    win_conditions = {
        "Town": "Eliminate all threats to the Town.",
        "Mafia": "Kill anyone that will not submit to the Mafia.",
        "Neutral": "Achieve your specific role objective.",
        "Coven": "Kill all who would oppose the Coven."
    }
    
    faction = info.get("faction", getattr(role, "faction", "Unknown"))
    win_condition = win_conditions.get(faction, "Achieve your role's specific objective.")
    
    # Role-specific details
    active_abilities = []
    passive_abilities = []
    
    # Add role-specific ability descriptions
    if role_name == "Doctor":
        active_abilities = ["NIGHT ABILITY: <heal>player</heal> - Heal one person each night, granting them Powerful defense. You may only heal yourself once. You will know if your target is attacked."]
        passive_abilities = ["Defense: None", "Visit Type: Non-harmful"]
    elif role_name == "Sheriff":
        active_abilities = ["NIGHT ABILITY: <investigate>player</investigate> - Investigate one player each night for suspicious activity. You will get 'Suspicious' or 'Not Suspicious' results."]
        passive_abilities = ["Defense: None", "Visit Type: Non-harmful"]
    elif role_name == "Jailor":
        active_abilities = [
            "DAY ABILITY: <jail>player</jail> - You can arrest a member of the town for interrogation at night.",
            "NIGHT ABILITY: <execute></execute> - Executes the person currently jailed, killing them. You have 3 executions. If you execute a town member you will lose the ability to execute."
        ]
        passive_abilities = ["Defense: None", "Jailed players cannot use abilities or vote"]
    elif role_name == "Bodyguard":
        active_abilities = ["NIGHT ABILITY: <protect>player</protect> - Protect another player each night. Die in their place if attacked while granting a Basic Defense vest to yourself once per game."]
        passive_abilities = ["Defense: None", "Counterattacks attackers"]
    elif role_name == "Vigilante":
        active_abilities = ["NIGHT ABILITY: <shoot>player</shoot> - Shoot a player at night (3 bullets). If you kill a town member, you will commit suicide the following night."]
        passive_abilities = ["Attack: Basic", "Defense: None"]
    elif role_name == "Veteran":
        active_abilities = ["NIGHT ABILITY: <alert></alert> - Go on alert, killing anyone who visits you (3 alerts max)."]
        passive_abilities = ["When on alert: Attack: Powerful, Defense: Basic"]
    # Add more role-specific descriptions as needed
    
    return RoleCard(
        name=role_name,
        faction=faction,
        alignment=info.get("alignment", getattr(role, "alignment", "Unknown")),
        win_condition=win_condition,
        passive_abilities=passive_abilities,
        active_abilities=active_abilities,
        attack=info.get("attack", getattr(role, "attack", "None")),
        defense=info.get("defense", getattr(role, "defense", "None")),
        visit_type=getattr(role, "visit_type", "Non-harmful"),
        immunities=getattr(role, "immunities", []),
        unique=info.get("is_unique", getattr(role, "is_unique", False))
    )


# ---------------------------------------------------------------------------
# Phase Information System
# ---------------------------------------------------------------------------

def get_phase_rules() -> str:
    """Get the complete phase rules text."""
    return """PHASE RULES (PLAIN TEXT, XML-FRIENDLY, AND ACTION-DRIVEN)
The game is divided into a sequence of phases. Your set of allowed actions and their effects depend entirely on the current phase, which will always be visible in your context. Each phase enables or restricts different interactions. Below are the phases and their rules:

Day Discussion:
All living players may use <speak> to communicate publicly, <whisper target="Player"> to privately message any player, or </wait> to take no action. All information tools and role abilities that are allowed during the day may be used at this time. Nominations and voting for trial are not available in this phase. You may update your will or view your will as needed.

Nomination:
All living players may continue to <speak>, <whisper target="Player">, or </wait>. During this phase, you may use <vote>Player</vote> to nominate a player to stand trial. Each player may only nominate one player per nomination cycle. Nominations may not be withdrawn after submission. Discussion continues during this phase. No voting for guilt or innocence occurs yet.

Defense:
Only the player who is currently on trial may use <speak>. All other players may not speak in the public channel but may still use <whisper target="Player"> or </wait>. No votes or nominations are permitted. Tools and abilities may be used if allowed by the game rules. The defense phase is solely for the accused player to address the town.

Judgement:
All living players except the player on trial may use <vote>guilty</vote>, <vote>innocent</vote>, or <vote>abstain</vote> to determine the outcome of the trial. The accused may not vote but can continue to <whisper target="Player"> or </wait>. Public discussion is closed; <speak> is not allowed for anyone during judgement. Tools and abilities may be used if permitted.

Last Words:
Only the player being executed may use <speak> to deliver a final message before their elimination. No other public messages are permitted. All players may still use <whisper target="Player"> or </wait>. The will cannot be updated during last words.

Pre-Night:
A brief transition phase; typically, no actions are required or allowed except </wait> if you wish to explicitly end your output.

Night:
All living players with night abilities may use their specific ability or action tag. Most roles cannot use <speak> during the night, but some special roles (such as Mafia, Coven, or other "faction chat" roles) may have access to a private chat channel and may use <speak>. <whisper target="Player"> is generally not available at night unless your role or the rules specify otherwise. You may update or view your will at night unless otherwise restricted. All information tools and abilities that are allowed at night may be used. Public chat is closed.

Game End:
When the game ends, you will be shown the victory conditions and the final result. No further actions are required."""


def get_real_time_warning() -> str:
    """Get the real-time play warning text."""
    return """Real-Time Play:
The game operates in real-time. This means that as you spend time thinking or generating your response, the game continues and other players may act, send messages, or perform actions during this period.

Each time you generate output, any events or changes in the environment that occurred while you were thinking will be injected into your context as new information. The longer your <think> block or your reasoning, the more time will pass in the game, and the more the environment may have changed by the time you are ready to act.

Therefore, you should balance the thoroughness of your reasoning with the need to respond in a timely manner, as excessive delays may cause you to miss important developments or opportunities. This applies to your entire generation time, not just thinking but also speaking for long amounts of time."""


# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------

def build_system_prompt(agent_name: str, role: 'Role', game: 'Game') -> str:
    """Build the complete static system prompt."""
    
    # Build role card
    role_card = build_role_card(role)
    
    # Get tools and interactions
    tools = discover_tools()
    interactions = get_available_interactions()
    
    # Build roster (static - updates only when structure changes)
    roster = [p.name for p in game.players if p.is_alive]
    
    # Game overview
    game_overview = f"""You are a participant in a game called Town of Salem. Town of Salem is an online social deduction game played between multiple participants, each of whom is randomly assigned a secret role belonging to a particular faction (e.g, Town, Mafia, Neutral, Coven). The game is structured in alternating night and day phases. During the day, players communicate publicly and vote to lynch others. During the night, some roles may perform actions. Your role and win condition are defined below. The rules, possible actions, and public information are available to you. Players may communicate, collaborate, deceive, or withhold information as they choose, according to their individual strategies and goals. Engage with the game as an ordinary player, utilizing your knowledge of the rules, your observations and your reasoning to make decisions. Do not reference being an AI or model. You are a player. When reasoning, planning, or gathering your internal thoughts, you must use the <think> and </think> tags. Everything inside <think> and </think> is private, internal reasoning and must never be revealed to other players. Use this space to analyze the situation, consider possibilities, or plan your next actions. You must complete your internal thought and close with </think> before using any tools or taking any action.

After finishing your internal reasoning with </think>, you may interact with the environment. This can include issuing public statements, casting votes, or using one of the tools available to you. You may only use tools after you have finished your internal thought. Never use a tool or act while inside a <think> block. Always separate your reasoning and your actions in this way."""
    
    # Internal reasoning protocol
    reasoning_protocol = """The following tools are available to you. Each tool must be invoked in the format provided, outside of the <think> block, and only after you have finished your thought. After you finish a <think> block and use a tool, the result of the tool will be provided to you as new information. Each time you receive new information from a tool or from the environment, you must start a new <think> block to process and reflect on the information you have just received. Only after reasoning in this way should you decide whether to use another tool, interact with the environment, or take no action. You must always alternate between thinking in <think> and </think> and then acting, never chaining together multiple tool uses or environment interactions in a single step. After each action, always wait for the outcome before thinking again and choosing your next move."""
    
    # Build system prompt sections
    sections = []
    
    # Header
    sections.append(f"Your name is {agent_name}. You are a {role_card.name}.")
    
    # Game overview
    sections.append(game_overview)
    
    # Role card
    sections.append(f"""Role: {role_card.name}
Alignment: {role_card.alignment}
Faction: {role_card.faction}
Win Condition: {role_card.win_condition}""")
    
    if role_card.passive_abilities:
        sections.append("Passive Abilities:\n" + "\n".join(f"• {ability}" for ability in role_card.passive_abilities))
    
    if role_card.active_abilities:
        sections.append("Active Abilities:\n" + "\n".join(f"• {ability}" for ability in role_card.active_abilities))
    
    # Internal reasoning protocol
    sections.append(reasoning_protocol)
    
    # Tools section
    tools_text = "The tools available to you are:\n"
    for tool in tools:
        tools_text += f"\nTool: {tool.syntax}\n"
        tools_text += f"Description: {tool.description}\n"
        tools_text += f"Example input: {tool.example_input}\n"
        tools_text += f"Example output: {tool.example_output}\n"
    
    sections.append(tools_text)
    
    # Interactions section
    interactions_text = "The interactions available to you are:\n"
    for interaction in interactions:
        interactions_text += f"\nInteraction: {interaction.syntax}\n"
        interactions_text += f"Description: {interaction.description}\n"
        interactions_text += f"Example input: {interaction.example_input}\n"
        interactions_text += f"Example output: {interaction.example_output}\n"
    
    sections.append(interactions_text)
    
    # Phase rules
    sections.append(get_phase_rules())
    
    # Real-time warning
    sections.append(get_real_time_warning())
    
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# User Prompt Builder
# ---------------------------------------------------------------------------

def build_user_prompt(game: 'Game', actor: 'Player') -> str:
    """Build the dynamic user prompt for the current turn."""
    
    sections = []
    
    # Phase and day information
    phase_name = game.phase.name.title() if hasattr(game, 'phase') else "Unknown"
    day_num = getattr(game, 'day', 1)
    sections.append(f"Phase: {game.time.name.title()} {day_num} - {phase_name}")
    
    # Alive roster
    alive_players = [p.name for p in game.players if p.is_alive]
    sections.append("Alive Roster\n" + "\n".join(f"- {name}" for name in alive_players))
    
    # Graveyard
    if hasattr(game, 'graveyard') and game.graveyard:
        graveyard_text = "Graveyard\n"
        for dead_player in game.graveyard:
            death_info = f"- {dead_player.name} ({dead_player.role.name.value})"
            if hasattr(dead_player, 'death_cause') and dead_player.death_cause:
                death_info += f" — {dead_player.death_cause}"
            if dead_player.last_will:
                death_info += f'. Last Will: "{dead_player.last_will}"'
            graveyard_text += death_info + "\n"
        sections.append(graveyard_text.rstrip())
    
    # Notebook (environment_terminal - always shown at top)
    if hasattr(actor, 'notebook') and actor.notebook:
        notebook_tokens = getattr(actor, 'notebook_tokens', 0)
        sections.append(f"Notebook (private, {notebook_tokens}/1500 tokens)\n{actor.notebook}")
    
    # Environment observations (placeholder for now)
    sections.append("Environment Observations\n[No new events since the phase began.]")
    
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Main Prompt Building Function
# ---------------------------------------------------------------------------

def build_complete_prompt(game: 'Game', actor: 'Player', model_name: str = "default") -> str:
    """Build the complete prompt for an agent using model-specific formatting."""
    
    # Get model configuration
    model_config = get_model_config(model_name)
    
    # Build system and user prompts
    system_prompt = build_system_prompt(actor.name, actor.role, game)
    user_prompt = build_user_prompt(game, actor)
    
    # Format according to model requirements
    return model_config.format_messages(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Legacy Compatibility Functions
# ---------------------------------------------------------------------------

def build_game_prompt(game: 'Game', actor: 'Player', observation: Optional[str] = None) -> str:
    """Legacy compatibility function."""
    return build_complete_prompt(game, actor)


def build_prompt(game: 'Game', actor: 'Player', observation: Optional[str] = None) -> str:
    """Legacy compatibility function."""
    return build_complete_prompt(game, actor) 