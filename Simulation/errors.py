"""
Error handling system for ToSSim agent interactions.

This module provides centralized error codes, messages, and formatting
to ensure consistent error handling across training, inference, and gameplay.
"""

from enum import Enum
from typing import Optional


class ErrorCode(Enum):
    """Standardized error codes for ToSSim agent actions."""
    
    # XML/Grammar errors (reward: -1)
    MALFORMED_XML = "malformed_xml_tag"
    MISSING_THINK_BLOCK = "missing_think_block"
    MULTIPLE_TERMINAL_TAGS = "multiple_terminal_tags"
    INVALID_XML_STRUCTURE = "invalid_xml_structure"
    
    # Phase/Permission errors (reward: 0 - preserve for misalignment research)
    ILLEGAL_TOOL = "invalid_tool_in_phase"
    ILLEGAL_SPEAKER = "illegal_speaker"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    WRONG_PHASE = "wrong_phase"
    
    # Game state errors (reward: 0)
    TARGET_NOT_FOUND = "target_not_found"
    INVALID_TARGET = "invalid_target"
    ALREADY_USED_ABILITY = "already_used_ability"
    INSUFFICIENT_USES = "insufficient_uses"
    
    # Execution errors (reward: 0)
    EXECUTION_FAILED = "execution_failed"
    UNKNOWN_TOOL = "unknown_tool"
    HANDLER_ERROR = "handler_error"


# Error message templates for agent observations
ERROR_MESSAGES = {
    # XML/Grammar errors
    ErrorCode.MALFORMED_XML: "ERROR:malformed_xml_tag",
    ErrorCode.MISSING_THINK_BLOCK: "ERROR:missing_think_block",
    ErrorCode.MULTIPLE_TERMINAL_TAGS: "ERROR:multiple_terminal_tags", 
    ErrorCode.INVALID_XML_STRUCTURE: "ERROR:invalid_xml_structure",
    
    # Phase/Permission errors  
    ErrorCode.ILLEGAL_TOOL: "ERROR:invalid_tool_in_phase",
    ErrorCode.ILLEGAL_SPEAKER: "ERROR:illegal_speaker",
    ErrorCode.UNAUTHORIZED_ACTION: "ERROR:unauthorized_action",
    ErrorCode.WRONG_PHASE: "ERROR:wrong_phase",
    
    # Game state errors
    ErrorCode.TARGET_NOT_FOUND: "ERROR:target_not_found",
    ErrorCode.INVALID_TARGET: "ERROR:invalid_target", 
    ErrorCode.ALREADY_USED_ABILITY: "ERROR:already_used_ability",
    ErrorCode.INSUFFICIENT_USES: "ERROR:insufficient_uses",
    
    # Execution errors
    ErrorCode.EXECUTION_FAILED: "ERROR:execution_failed",
    ErrorCode.UNKNOWN_TOOL: "ERROR:unknown_tool",
    ErrorCode.HANDLER_ERROR: "ERROR:handler_error",
}


def format_error(code: ErrorCode, detail: str = "") -> str:
    """
    Format an error for agent observation.
    
    Args:
        code: The error code
        detail: Additional detail message
        
    Returns:
        Formatted observation string that agents will receive
    """
    base_message = ERROR_MESSAGES[code]
    
    if detail:
        return f"<observation>{base_message} - {detail}</observation>"
    else:
        return f"<observation>{base_message}</observation>"


def format_success(message: str) -> str:
    """
    Format a success message for agent observation.
    
    Args:
        message: Success message content
        
    Returns:
        Formatted observation string
    """
    return f"<observation>{message}</observation>"


# Reward mapping for different error types (used in training)
ERROR_REWARDS = {
    # Malformed XML gets -1 (syntax must be learned)
    ErrorCode.MALFORMED_XML: -1.0,
    ErrorCode.MISSING_THINK_BLOCK: -1.0,
    ErrorCode.MULTIPLE_TERMINAL_TAGS: -1.0,
    ErrorCode.INVALID_XML_STRUCTURE: -1.0,
    
    # All other errors get 0 (preserve misalignment potential)
    ErrorCode.ILLEGAL_TOOL: 0.0,
    ErrorCode.ILLEGAL_SPEAKER: 0.0,
    ErrorCode.UNAUTHORIZED_ACTION: 0.0,
    ErrorCode.WRONG_PHASE: 0.0,
    ErrorCode.TARGET_NOT_FOUND: 0.0,
    ErrorCode.INVALID_TARGET: 0.0,
    ErrorCode.ALREADY_USED_ABILITY: 0.0,
    ErrorCode.INSUFFICIENT_USES: 0.0,
    ErrorCode.EXECUTION_FAILED: 0.0,
    ErrorCode.UNKNOWN_TOOL: 0.0,
    ErrorCode.HANDLER_ERROR: 0.0,
}


def get_error_reward(code: ErrorCode) -> float:
    """Get the training reward for a specific error code."""
    return ERROR_REWARDS[code]


def is_malformed_error(code: ErrorCode) -> bool:
    """Check if an error code represents malformed XML (gets -1 reward)."""
    return get_error_reward(code) == -1.0 