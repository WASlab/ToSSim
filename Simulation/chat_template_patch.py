from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import inspect
import logging

import transformers
from transformers import PreTrainedTokenizerBase

__all__ = [
    "format_chat",
]

def _tokenizer_has_apply_chat_template(tok: PreTrainedTokenizerBase) -> bool:
    """Return True if *tokenizer* implements the HF v4.39 chat template API."""
    return hasattr(tok, "apply_chat_template") and callable(tok.apply_chat_template)


def _can_use_builtin_template(tok: PreTrainedTokenizerBase, roles: List[str]) -> bool:
    """Heuristic: try a dry‑run of apply_chat_template with the supplied roles.

    Some vocabularies expose the method but don’t include *all* roles we need.
    We do a safe probe (catching ``KeyError``) and fall back if the probe fails.
    """
    if not _tokenizer_has_apply_chat_template(tok):
        return False
    probe_msgs = [{"role": r, "content": "x"} for r in roles]
    try:
        _ = tok.apply_chat_template(probe_msgs, tokenize=False)
        return True
    except KeyError:
        return False
    except Exception as exc:
        logging.debug("chat_template probe failed: %s", exc)
        return False


def _gemma_manual_format(system: str, user: str, observation: Optional[str]) -> Tuple[List[Dict[str, str]], str]:
    """Return Gemma‑style (<start_of_turn>) conversation and the raw text."""
    msgs: List[Dict[str, str]] = []
    # Gemma treats system prompt as *user* role.
    msgs.append({"role": "user", "content": system})
    if observation is not None:
        msgs.append({"role": "observation", "content": observation})
    msgs.append({"role": "user", "content": user})

    parts: List[str] = []
    for m in msgs:
        r = m["role"]
        parts.append(f"<start_of_turn>{r}\n{m['content']}<end_of_turn>")
    parts.append("<start_of_turn>model\n")  # generation starts here
    return msgs, "\n".join(parts)


def format_chat(
    tokenizer: PreTrainedTokenizerBase,
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    observation: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> Tuple[List[Dict[str, str]], str]:
    """Return (messages, rendered_text) following the best available template.

    ``messages`` is always a list[dict] suitable for apply_chat_template.
    ``rendered_text`` is what you’d actually feed into ``model(**tok(...))`` if
    you’re *not* using ``apply_chat_template`` (e.g. for Gemma fallback).
    """

    roles_order = ["system", "user"]
    if observation is not None:
        roles_order.append("observation")
    roles_order.append("assistant")

    # 1) Try HF built‑ins --------------------------------------------------
    if _can_use_builtin_template(tokenizer, roles_order):
        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if observation is not None:
            msgs.append({"role": "observation", "content": observation})
        msgs.append({"role": "user", "content": user_prompt})
        # HF will append the generation role when add_generation_prompt=True
        rendered = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        return msgs, rendered

    # 2) Gemma manual or generic fallback ---------------------------------
    lower_name = model_name.lower()
    if "gemma" in lower_name:
        return _gemma_manual_format(system_prompt, user_prompt, observation)

    # Generic fallback (<|system|> / <|user|> / <|assistant|>)
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    if observation is not None:
        msgs.append({"role": "user", "content": observation})  # no obs role => user
    msgs.append({"role": "user", "content": user_prompt})

    text_parts = [
        f"<|system|>\n{system_prompt}</s>",
    ]
    if observation is not None:
        text_parts.append(f"<|user|>\n{observation}</s>")
    text_parts.append(f"<|user|>\n{user_prompt}</s>")
    if add_generation_prompt:
        text_parts.append("<|assistant|>\n")
    rendered = "\n".join(text_parts)
    return msgs, rendered