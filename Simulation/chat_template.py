"""
Chat template system for different model formats.

This module provides a clean way to format prompts for different models
while supporting the observation role that's not standard in most chat templates.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Supported model types and their chat templates."""
    GEMMA = "gemma"
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    DEFAULT = "default"


@dataclass
class ChatMessage:
    """A single chat message with role and content."""
    role: str
    content: str


def apply_chat_template(
    messages: List[ChatMessage], 
    model_type: ModelType = ModelType.GEMMA,
    add_generation_prompt: bool = True
) -> str:
    """
    Apply the appropriate chat template for the given model type.
    
    Args:
        messages: List of chat messages with role and content
        model_type: The model type to format for
        add_generation_prompt: Whether to add the assistant prompt at the end
    
    Returns:
        Formatted prompt string
    """
    
    if model_type == ModelType.GEMMA:
        return _apply_gemma_template(messages, add_generation_prompt)
    elif model_type == ModelType.LLAMA:
        return _apply_llama_template(messages, add_generation_prompt)
    elif model_type == ModelType.MISTRAL:
        return _apply_mistral_template(messages, add_generation_prompt)
    elif model_type == ModelType.DEEPSEEK:
        return _apply_deepseek_template(messages, add_generation_prompt)
    else:
        return _apply_default_template(messages, add_generation_prompt)


def _apply_gemma_template(messages: List[ChatMessage], add_generation_prompt: bool) -> str:
    """Apply Gemma chat template."""
    formatted_parts = []
    
    for msg in messages:
        if msg.role == "system":
            formatted_parts.append(f"<start_of_turn>user\n{msg.content}")
        elif msg.role == "user":
            formatted_parts.append(f"<start_of_turn>user\n{msg.content}")
        elif msg.role == "observation":
            formatted_parts.append(f"<start_of_turn>observation\n{msg.content}")
        elif msg.role == "assistant":
            formatted_parts.append(f"<start_of_turn>assistant\n{msg.content}")
    
    if add_generation_prompt:
        formatted_parts.append("<start_of_turn>assistant\n")
    
    return "\n\n".join(formatted_parts)


def _apply_llama_template(messages: List[ChatMessage], add_generation_prompt: bool) -> str:
    """Apply Llama chat template."""
    formatted_parts = []
    
    for msg in messages:
        if msg.role == "system":
            formatted_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{msg.content}<|eot_id|>")
        elif msg.role == "user":
            formatted_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{msg.content}<|eot_id|>")
        elif msg.role == "observation":
            # Llama doesn't have observation token, so we use user with observation content
            formatted_parts.append(f"<|start_header_id|>user<|end_header_id|>\n<observation>{msg.content}</observation><|eot_id|>")
        elif msg.role == "assistant":
            formatted_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{msg.content}<|eot_id|>")
    
    if add_generation_prompt:
        formatted_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    
    return "".join(formatted_parts)


def _apply_mistral_template(messages: List[ChatMessage], add_generation_prompt: bool) -> str:
    """Apply Mistral chat template."""
    formatted_parts = []
    
    for msg in messages:
        if msg.role == "system":
            formatted_parts.append(f"<s>[INST] {msg.content} [/INST]")
        elif msg.role == "user":
            formatted_parts.append(f"[INST] {msg.content} [/INST]")
        elif msg.role == "observation":
            # Mistral doesn't have observation, so we use user with observation content
            formatted_parts.append(f"[INST] <observation>{msg.content}</observation> [/INST]")
        elif msg.role == "assistant":
            formatted_parts.append(f"{msg.content}</s>")
    
    if add_generation_prompt:
        formatted_parts.append("")
    
    return "".join(formatted_parts)


def _apply_deepseek_template(messages: List[ChatMessage], add_generation_prompt: bool) -> str:
    """Apply DeepSeek chat template."""
    formatted_parts = []
    
    for msg in messages:
        if msg.role == "system":
            formatted_parts.append(f"{{{{- if .System }}}}{{msg.content}}{{{{ end }}}}")
        elif msg.role == "user":
            formatted_parts.append(f"{{{{- if eq .Role \"user\" }}}}REDACTED_SPECIAL_TOKEN{{{{ .Content }}}}")
        elif msg.role == "observation":
            # DeepSeek doesn't have observation, so we use user with observation content
            formatted_parts.append(f"{{{{- if eq .Role \"user\" }}}}REDACTED_SPECIAL_TOKEN<observation>{{{{ .Content }}}}</observation>")
        elif msg.role == "assistant":
            formatted_parts.append(f"{{{{- else if eq .Role \"assistant\" }}}}REDACTED_SPECIAL_TOKEN{{{{ .Content }}}}{{{{- if not $last }}}}REDACTED_SPECIAL_TOKEN{{{{- end }}}}")
    
    if add_generation_prompt:
        formatted_parts.append("REDACTED_SPECIAL_TOKEN")
    
    return "".join(formatted_parts)


def _apply_default_template(messages: List[ChatMessage], add_generation_prompt: bool) -> str:
    """Apply default chat template (similar to Gemma but with standard tokens)."""
    formatted_parts = []
    
    for msg in messages:
        if msg.role == "system":
            formatted_parts.append(f"<|system|>\n{msg.content}")
        elif msg.role == "user":
            formatted_parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "observation":
            formatted_parts.append(f"<observation>\n{msg.content}\n</observation>")
        elif msg.role == "assistant":
            formatted_parts.append(f"<|assistant|>\n{msg.content}")
    
    if add_generation_prompt:
        formatted_parts.append("<|assistant|>\n")
    
    return "\n\n".join(formatted_parts)


def build_game_messages(
    system_prompt: str,
    user_prompt: str,
    observation: Optional[str] = None,
    model_type: ModelType = ModelType.GEMMA
) -> str:
    """
    Build a complete game prompt using the appropriate chat template.
    
    Args:
        system_prompt: The system prompt content
        user_prompt: The user prompt content
        observation: Optional observation content
        model_type: The model type to format for
    
    Returns:
        Formatted prompt string
    """
    
    messages = [
        ChatMessage("system", system_prompt),
        ChatMessage("user", user_prompt)
    ]
    
    if observation:
        messages.append(ChatMessage("observation", observation))
    
    return apply_chat_template(messages, model_type, add_generation_prompt=True) 