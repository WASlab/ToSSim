"""Integration test that drives the real ``MatchRunner`` using a single
DeepSpeed model for all agents.

This exercises the full ToSSim stack – prompt building, tool routing and the
game logic – by having one tiny model play every role in the default lobby.
"""

import sys
import types
import pytest
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import jinja2
from pathlib import Path
from typing import List, Dict
from runner.lobby_loader import load_lobby
from inference.client import register_local
# ---------------------------------------------------------------------------
# Import MatchRunner without triggering NVML initialisation.
# ---------------------------------------------------------------------------
pynvml_stub = types.SimpleNamespace(
    nvmlInit=lambda: None,
    nvmlDeviceGetCount=lambda: 0,
    nvmlDeviceGetHandleByIndex=lambda idx: None,
    nvmlDeviceGetName=lambda handle: b"GPU",
    nvmlDeviceGetMemoryInfo=lambda handle: types.SimpleNamespace(total=0, free=0),
)
sys.modules["pynvml"] = pynvml_stub

from runner.match_runner import MatchRunner


class DSChatClient:
    """Very small DeepSpeed-backed client compatible with ``InferenceClient``."""

    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = deepspeed.init_inference(
            model,
            mp_size=1,
            
            dtype=torch.float32,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )
        self.model.eval()

    def _apply_chat_template(self, messages: List[Dict[str, str]], model_type: str = "gemma") -> str:
        template_dir = Path(__file__).parent.parent / "inference" / "templates"
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=False)

        if model_type.lower() == "gemma":
            template = env.get_template("gemma_chat_template.jinja")
        else:
            template = env.get_template("chat_template.jinja")

        return template.render(messages=messages)

    def chat(self, messages, **_):
        # Use the _apply_chat_template method to format the prompt
        prompt = self._apply_chat_template(messages)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = self.model.generate(input_ids, max_new_tokens=8)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Wrap in <speak> tags so the game accepts it as a valid action.
        return {"choices": [{"message": {"content": f"{text}"}}]}

class DeepSpeedEngine:
    """
    Minimal engine that uses 🤗 pipeline() + DeepSpeed‑FastGen.
    Compatible with MatchRunner's register/release API.
    """
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        # --- load + optimise model ------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # DeepSpeed‑FastGen kernel injection (single‑GPU path)
        ds_engine = deepspeed.init_inference(
            base_model,
            mp_size=1,                                 # tensor‑parallel degree
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )
        optimised_model = ds_engine.module            # unwrap to raw nn.Module

        # Wrap everything in an HF pipeline (device -1 = CPU fallback)
        self.pipe = pipeline(
            "text-generation",
            model=optimised_model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Jinja env for your chat templates
        tmpl_dir = Path(__file__).parent.parent / "inference" / "templates"
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(tmpl_dir), autoescape=False
        )

    # --------------------------------------------------------------------- helpers
    def _render_prompt(
        self, messages: List[Dict[str, str]], model_type: str = "gemma"
    ) -> str:
        tmpl_file = (
            "gemma_chat_template.jinja" if model_type.lower() == "gemma"
            else "chat_template.jinja"
        )
        return self._env.get_template(tmpl_file).render(messages=messages)

    # ---------------------------------------------------------------- public API
    def chat(self, messages, **_):
        prompt = self._render_prompt(messages)
        # pipeline always appends prompt to its own output → strip it back off
        full = self.pipe(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            return_full_text=True,
        )[0]["generated_text"]
        full = self.pipe(messages=messages, max_new_tokens=8, do_sample=False, **kw)[0]
        return {"choices": [{"message": {"content": full["generated_text"]}}]}

    # ---- MatchRunner integration ----------------------------------------------
    def register_agent(self, aid: str, model_name: str):
        local_url = f"local://{aid}"
        register_local(local_url, lambda msgs, **kw: self.chat(msgs, model_name, **kw))
        return (0, local_url)  

    def release_agent(self, aid: str):  # noqa: D401
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepSpeed FastGen requires CUDA")
def test_finally_a_some_good_tests():
    """Run a short day/night cycle with every player handled by one model."""

    lobby = load_lobby()  # 15-player default lobby
    engine = DeepSpeedEngine()

    runner = MatchRunner(engine, lobby)

    # One day and one night cycle is enough to exercise the stack.
    runner._process_day_phase()
    if not runner.game.game_is_over():
        runner._process_night_phase()

    # Every player should have produced at least one message
    for ctx in runner.agents.values():
        assert ctx.chat_history, f"no output from {ctx.player.name}"
        


