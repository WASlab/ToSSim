"""Simulation.tokenizer_utils – centralized token counting utilities.

This module provides token counting functionality using Gemma-3's tokenizer
for accurate token measurement across the simulation system.
"""

from __future__ import annotations
from typing import Optional
import os

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

# Global tokenizer instance
_tokenizer: Optional[object] = None


def get_tokenizer():
    """Get the global Gemma-3 tokenizer instance."""
    global _tokenizer
    
    if _tokenizer is None:
        if not TOKENIZER_AVAILABLE:
            raise ImportError("transformers library not available for tokenization")
        
        # Use Gemma-3 model for tokenization as specified
        model_name = "google/gemma-3-4b-it"  # Use 3-4b-it for consistency
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using Gemma-3 tokenizer.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens in the text
        
    Raises:
        ImportError: If transformers library is not available
    """
    if not text:
        return 0
    
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback to rough approximation if tokenizer fails
        return len(text) // 4 + 1


def truncate_to_token_limit(text: str, max_tokens: int) -> tuple[str, int]:
    """Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Tuple of (truncated_text, actual_token_count)
    """
    if not text:
        return "", 0
    
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text, len(tokens)
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text, len(truncated_tokens)
        
    except Exception:
        # Fallback to character-based truncation
        estimated_chars = max_tokens * 4
        if len(text) <= estimated_chars:
            return text, len(text) // 4 + 1
        
        truncated = text[:estimated_chars]
        return truncated, max_tokens


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimation fallback (1 token ≈ 4 characters).
    
    This is used as a fallback when the tokenizer is not available
    or for quick approximations.
    """
    if not text:
        return 0
    return len(text) // 4 + 1


def remove_tokens_from_start(text: str, tokens_to_remove: int) -> tuple[str, int]:
    """Remove approximately N tokens from the start of text (for FIFO).
    
    Args:
        text: Text to remove tokens from
        tokens_to_remove: Number of tokens to remove from start
        
    Returns:
        Tuple of (remaining_text, tokens_actually_removed)
    """
    if not text or tokens_to_remove <= 0:
        return text, 0
    
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if tokens_to_remove >= len(tokens):
            return "", len(tokens)
        
        # Remove tokens from start
        remaining_tokens = tokens[tokens_to_remove:]
        remaining_text = tokenizer.decode(remaining_tokens, skip_special_tokens=True)
        
        return remaining_text, tokens_to_remove
        
    except Exception:
        # Fallback: remove roughly equivalent characters
        chars_to_remove = tokens_to_remove * 4
        if chars_to_remove >= len(text):
            return "", estimate_tokens_rough(text)
        
        remaining_text = text[chars_to_remove:]
        return remaining_text, tokens_to_remove 