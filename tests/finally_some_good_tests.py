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
from transformers import AutoModelForCausalLM, AutoTokenizer
import jinja2
from pathlib import Path
from typing import List, Dict
from runner.lobby_loader import load_lobby

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
    """Minimal engine implementing the ``register_agent`` API."""

    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        self.client = DSChatClient(model_name)

    def register_agent(self, aid: str, model: str):
        # MatchRunner expects a (lane_id, url) tuple.
        return (0, "http://localhost:8000")

    def release_agent(self, aid: str):
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
        


