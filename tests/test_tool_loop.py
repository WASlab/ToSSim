import json
from Simulation.tools.registry import execute_tool
from inference.tool_router import apply_first_tool_call

# ---------------------------------------------------------------------------
# Lightweight fake client
# ---------------------------------------------------------------------------

class DummyClient:
    """Cycles through pre-defined assistant messages."""

    def __init__(self, responses):
        self._responses = responses
        self.index = 0

    def chat(self, *_, **__):  # signature compatible with InferenceClient.chat
        assert self.index < len(self._responses), "No more dummy responses"
        content = self._responses[self.index]
        self.index += 1
        return {"choices": [{"message": {"content": content}}]}


# ---------------------------------------------------------------------------
# Tool-loop helper identical to what MatchRunner will implement
# ---------------------------------------------------------------------------

def run_tool_loop(client, initial_messages):
    """Return (patched_transcript, observation_list) until a terminal tag seen."""
    msgs = list(initial_messages)
    observations = []
    while True:
        resp = client.chat(msgs)
        assistant_text = resp["choices"][0]["message"]["content"]
        patched, obs = apply_first_tool_call(assistant_text)
        msgs.append({"role": "assistant", "content": patched})
        if obs is not None:
            observations.append(obs)
            msgs.append({"role": "user", "content": f"<observation>{obs}</observation>"})
            continue  # let the assistant read the observation
        # Stop when a terminal public tag appears
        if any(tag in patched for tag in ("<speak>", "<vote>", "<wait>", "<whisper>")):
            break
    return msgs, observations


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

def test_tool_loop_single_get_role():
    assistant_responses = [
        "<think>I should look up the Bodyguard.<get_role>Bodyguard</get_role>",
        "<think>Now I know.<speak>Hello town!</speak>"
    ]
    client = DummyClient(assistant_responses)
    transcript, obs = run_tool_loop(client, [])

    # The loop should have executed the get_role tool exactly once
    assert len(obs) == 1
    obs_json = json.loads(obs[0])
    assert obs_json["role_info"]["name"] == "Bodyguard"
    # Final assistant message ends with <speak>
    assert transcript[-1]["role"] == "assistant"
    assert "<speak>" in transcript[-1]["content"]

    # The loop should have executed the get_role tool exactly once
    assert len(obs) == 1
    obs_json = json.loads(obs[0])
    assert obs_json["role_info"]["name"] == "Bodyguard"
    # Final assistant message ends with <speak>
    assert transcript[-1]["role"] == "assistant"
    assert "<speak>" in transcript[-1]["content"] 