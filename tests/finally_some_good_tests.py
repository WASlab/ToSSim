"""
Integration test that drives the real ``MatchRunner`` using a single
DeepSpeed model for all agents.

This exercises the full ToSSim stack – prompt building, tool routing and the
game logic – by having one tiny model play every role in the default lobby.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Dict, List
import tempfile

import deepspeed
import jinja2
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from inference.client import register_local
from runner.lobby_loader import load_lobby
from Simulation.event_logger import GameLogger
import sys
import os

# ---------------------------------------------------------------------------
# Import MatchRunner *without* triggering NVML initialisation (important for
# headless CI environments with no visible GPU devices).
# ---------------------------------------------------------------------------
pynvml_stub = types.SimpleNamespace(
    nvmlInit=lambda: None,
    nvmlDeviceGetCount=lambda: 0,
    nvmlDeviceGetHandleByIndex=lambda idx: None,
    nvmlDeviceGetName=lambda handle: b"GPU",
    nvmlDeviceGetMemoryInfo=lambda handle: types.SimpleNamespace(total=0, free=0),
)
sys.modules["pynvml"] = pynvml_stub

from runner.match_runner import MatchRunner  # noqa: E402  (import after stubbing)


# ---------------------------------------------------------------------------


class DeepSpeedEngine:
    """
    Lightweight engine that keeps inference fully in‑process:

    * loads a tiny GPT‑2 model
    * injects DeepSpeed FastGen kernels
    * wraps everything in 🤗 `pipeline("text-generation")`
    * exposes a `chat()` method that MatchRunner can call via the
      `local://<agent_id>` transport.
    """

    def __init__(self, model_name: str = "google/gemma-3-27b-it"):
        # --- model + tokenizer --------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # DeepSpeed FastGen kernel injection
        ds_engine = deepspeed.init_inference(
            base_model,
            mp_size=1,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )
        optimised_model = ds_engine.module  # unwrap to nn.Module

        # Hugging Face pipeline wrapper (device −1 = CPU fallback)
        self.pipe = pipeline(
            task="text-generation",
            model=optimised_model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Jinja2 environment for chat templates
        tmpl_dir = Path(__file__).parent.parent / "inference" / "templates"
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(tmpl_dir),
            autoescape=False,
        )

    # --------------------------------------------------------------------- helpers

    def _render_prompt(
        self,
        messages: List[Dict[str, str]],
        model_type: str = "gemma",
    ) -> str:
        tmpl_file = (
            "gemma_chat_template.jinja"
            if model_type.lower() == "gemma"
            else "chat_template.jinja"
        )
        template = self._env.get_template(tmpl_file)
        return template.render(messages=messages)

    # ---------------------------------------------------------------- public API

    def chat(self, messages: List[Dict[str, str]], **kw) -> Dict:
        """MatchRunner → InferenceClient fast‑path entry."""
        prompt = self._render_prompt(messages)
        generated = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            return_full_text=True,
        )[0]["generated_text"]
        reply = generated[len(prompt) :].lstrip()
        print(reply)
        return {"choices": [{"message": {"content": reply}}]}

    # ---------------------------------------------------------------- MatchRunner integration

    def register_agent(self, aid: str, _model_name: str):
        """
        Provide MatchRunner with a *local* endpoint instead of HTTP.

        `InferenceClient` looks up handlers registered via `register_local`
        and calls them directly, bypassing the network stack entirely.
        """
        local_url = f"local://{aid}"
        register_local(local_url, self.chat)
        # lane_id == 0 because we do not shard across GPUs in this test
        return 0, local_url

    def release_agent(self, _aid: str):  # noqa: D401
        """Nothing to clean up in the single‑process test harness."""
        return


# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DeepSpeed FastGen requires CUDA"
)
def test_finally_a_some_good_tests() -> None:
    """
    Run a short day/night cycle with every player handled by the same model.

    The goal is *not* to win the game but to exercise the entire ToSSim stack
    end‑to‑end, ensuring that prompt routing, tool calls and the game loop all
    execute without error.
    """
    lobby = load_lobby()  # 15‑player default lobby
    engine = DeepSpeedEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        game_logger = GameLogger(game_id="test_game", log_dir=Path(tmpdir))
        runner = MatchRunner(engine, lobby, game_logger=game_logger)

        # One day and one night cycle is enough to hit every code path.
        runner._process_day_phase()
        if not runner.game.game_is_over():
            runner._process_night_phase()

    # Assert each agent produced at least one message.
    for ctx in runner.agents.values():
        assert ctx.chat_history, f"no output from {ctx.player.name}"
        
if __name__ == "__main__":
    test_finally_a_some_good_tests()