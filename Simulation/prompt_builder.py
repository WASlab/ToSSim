"""
Prompt building facade for ToSSim agents.

This module provides a single entry point for building prompts that:
1. Uses the existing inference.templates.prompt_builder under the hood
2. Adds game-specific context (phase, graveyard, etc.)
3. Includes phase-specific system hints
4. Handles observation formatting consistently

All consumers (training, inference, self-play) should use this module
to ensure consistent prompt formatting across the codebase.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

# Import the existing prompt builder
try:
    from inference.templates.prompt_builder import build_chat_messages as _build_chat_messages
    from inference.templates.prompt_builder import build_role_card, generate_agent_name
except ImportError:
    # Fallback for cases where inference module isn't available
    def _build_chat_messages(role, public_state, observation, history, **kwargs):
        return [{"role": "system", "content": "ToSSim Agent"}, {"role": "user", "content": "Game state"}]
    
    def build_role_card(role):
        return type('RoleCard', (), {'name': str(role.name.value) if hasattr(role, 'name') else 'Unknown'})()
    
    def generate_agent_name():
        return "Agent"

from .enums import Time, Phase
from .errors import format_success

if TYPE_CHECKING:
    from .game import Game
    from .player import Player


def build_game_prompt(game: 'Game', actor: 'Player', observation: Optional[str] = None) -> str:
    """
    Build a complete prompt for an agent in the current game state.
    
    Args:
        game: Current game instance
        actor: Player to build prompt for
        observation: Recent observation/result from last action
        
    Returns:
        Formatted prompt string ready for model input
    """
    
    # Build public game state
    public_state = _build_public_state(game)
    
    # Get chat history for this player (simplified for now)
    history = _build_chat_history(game, actor)
    
    # Add phase-specific observation prefix
    if observation is None:
        observation = _build_phase_observation(game)
    
    # Use existing prompt builder
    messages = _build_chat_messages(
        role=actor.role,
        public_state=public_state,
        observation=observation,
        history=history,
        observation_role="system"
    )
    
    # Convert messages to single prompt string
    return _messages_to_prompt_string(messages)


def _build_public_state(game: 'Game') -> Dict[str, Any]:
    """Build public game state dictionary."""
    
    alive_players = [p for p in game.players if p.is_alive]
    
    state = {
        "day": getattr(game, "day", 1),
        "time": game.time.name.lower() if hasattr(game, 'time') else "day",
        "phase": game.phase.name.lower() if hasattr(game, 'phase') else "discussion",
        "alive_count": len(alive_players),
        "alive_players": [p.name for p in alive_players],
        "graveyard": [],
        "vote_board": [],
        "trials_today": 0
    }
    
    # Add graveyard info
    if hasattr(game, 'graveyard') and game.graveyard:
        state["graveyard"] = [
            {
                "name": p.name, 
                "role": p.role.name.value if hasattr(p.role, 'name') else "Unknown",
                "death_cause": getattr(p, 'death_cause', 'Unknown')
            }
            for p in game.graveyard
        ]
    
    # Add day phase specific info
    if hasattr(game, 'day_phase_manager') and game.day_phase_manager:
        dpm = game.day_phase_manager
        
        # Voting info
        if hasattr(dpm, 'nominations') and dpm.nominations:
            vote_board = []
            for nominee, voters in dpm.nominations.items():
                vote_board.append({
                    "nominee": nominee.name,
                    "voters": [v.name for v in voters],
                    "vote_count": len(voters)
                })
            state["vote_board"] = vote_board
        
        # Trial info
        if hasattr(dpm, 'on_trial') and dpm.on_trial:
            state["on_trial"] = dpm.on_trial.name
        
        if hasattr(dpm, 'trials_remaining'):
            state["trials_today"] = 3 - dpm.trials_remaining
    
    return state


def _build_chat_history(game: 'Game', actor: 'Player') -> List[Dict[str, Any]]:
    """Build chat history for the player (simplified for now)."""
    
    # For now, return minimal history
    # In a full implementation, this would include recent chat messages,
    # voting history, etc.
    
    history = [
        {
            "agent_name": actor.name,
            "role": "system",
            "content": f"You are {actor.name} in this Town of Salem game."
        }
    ]
    
    return history


def _build_phase_observation(game: 'Game') -> str:
    """Build phase-specific observation prefix."""
    
    if not hasattr(game, 'time') or not hasattr(game, 'phase'):
        return "<observation>Game has started.</observation>"
    
    time_name = game.time.name.lower()
    phase_name = game.phase.name.lower()
    
    # Phase-specific hints and information
    if game.time == Time.DAY:
        if game.phase == Phase.DISCUSSION:
            return "<observation>Day discussion phase. You may speak freely and use information tools.</observation>"
        elif game.phase == Phase.NOMINATION:
            return "<observation>Nomination phase. You may speak and vote to nominate players for trial.</observation>"
        elif game.phase == Phase.DEFENSE:
            on_trial = "Unknown"
            if hasattr(game, 'day_phase_manager') and game.day_phase_manager and hasattr(game.day_phase_manager, 'on_trial'):
                on_trial = game.day_phase_manager.on_trial.name if game.day_phase_manager.on_trial else "Unknown"
            return f"<observation>Defense phase. {on_trial} is on trial and may defend themselves.</observation>"
        elif game.phase == Phase.JUDGEMENT:
            on_trial = "Unknown"
            if hasattr(game, 'day_phase_manager') and game.day_phase_manager and hasattr(game.day_phase_manager, 'on_trial'):
                on_trial = game.day_phase_manager.on_trial.name if game.day_phase_manager.on_trial else "Unknown"
            return f"<observation>Judgement phase. Vote guilty or innocent for {on_trial}.</observation>"
        else:
            return f"<observation>Day phase: {phase_name}.</observation>"
    
    elif game.time == Time.NIGHT:
        return "<observation>Night phase. Use your night abilities and information tools. No public speaking.</observation>"
    
    else:
        return f"<observation>{time_name} {phase_name} phase.</observation>"


def _messages_to_prompt_string(messages: List[Dict[str, Any]]) -> str:
    """Convert message list to a single prompt string."""
    
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"Scenario: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Previous: {content}")
    
    prompt_parts.append("Response:")
    return "\n\n".join(prompt_parts)


def build_training_prompt(game: 'Game', actor: 'Player') -> str:
    """
    Build a prompt specifically for training scenarios.
    
    This is a simplified version that focuses on the core format
    needed for Dr GRPO training.
    """
    
    # Build basic game context
    public_state = _build_public_state(game)
    phase_obs = _build_phase_observation(game)
    
    # Create training scenario description
    scenario_parts = [
        f"You are {actor.name}, a {actor.role.name.value if hasattr(actor.role, 'name') else 'Player'}.",
        f"Current phase: {public_state['time']} {public_state['phase']}",
        f"Day {public_state['day']}, {public_state['alive_count']} players alive"
    ]
    
    if public_state['graveyard']:
        graveyard_summary = ", ".join([f"{p['name']} ({p['role']})" for p in public_state['graveyard']])
        scenario_parts.append(f"Graveyard: {graveyard_summary}")
    
    if public_state['vote_board']:
        vote_summary = ", ".join([f"{v['nominee']} ({v['vote_count']} votes)" for v in public_state['vote_board']])
        scenario_parts.append(f"Current nominations: {vote_summary}")
    
    if 'on_trial' in public_state:
        scenario_parts.append(f"{public_state['on_trial']} is on trial")
    
    # System prompt for training
    system_prompt = """You are an AI agent playing Town of Salem. You must respond in the exact format:

<think>
Your reasoning here...
</think>
<ACTION_TAG>content if needed</ACTION_TAG>

RULES:
- Always start with exactly one <think> block
- End with exactly one action or tool tag
- Tool tags: <get_role>, <chat_history>, <graveyard>, <check_will>, <view_will>
- Action tags: <speak>, <wait>, <vote>, <kill>, <protect>, <investigate>, etc.
- No extra text outside the required format"""
    
    scenario = " ".join(scenario_parts)
    
    return f"System: {system_prompt}\n\nScenario: {scenario}\n\n{phase_obs}\n\nResponse:"


# Convenience function for backward compatibility
def build_prompt(game: 'Game', actor: 'Player', observation: Optional[str] = None) -> str:
    """Alias for build_game_prompt for backward compatibility."""
    return build_game_prompt(game, actor, observation) 