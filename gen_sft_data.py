"""
SFT dataset generator

Features
========
- Converts game trace logs into supervised fine-tuning (SFT) training samples
- Generates prompt/completion pairs for every agent turn in a chat-style format
- Adds metadata (e.g., agent ID, turn number, role, game ID) for filtering and analysis
- Supports both single and multi-game JSON input formats
- Outputs OpenAI-style chat format using "messages" list with user/assistant roles
- One sample per agent turn (optionally with partial completions)

Usage
=====
    python generate_sft_dataset.py input_traces.json output_dataset.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

def generate_prompt(turn: Dict) -> str:
    """Constructs a user prompt from a single turn."""
    return (
        f"Turn {turn['turn']}.\n"
        f"You are the {turn['role']}.\n"
        f"Observation: {turn['obs']}\n"
        f"What is your next action?"
    )


# Helper: Sample partial completions
from typing import List
def sample_partial_completions(completion: str) -> List[str]:
    """Returns truncated versions of the action to simulate incomplete messages."""
    tokens = completion.split()
    # Use two partial samples: first third and first two-thirds
    one_third = " ".join(tokens[:max(1, len(tokens) // 3)])
    two_thirds = " ".join(tokens[:max(1, 2 * len(tokens) // 3)])
    return [one_third, two_thirds] if len(tokens) > 3 else []


def convert_trace_to_sft(trace: Dict, include_partial: bool = False) -> List[Dict]:
    """Converts a single game trace into SFT-style prompt/completion pairs."""
    sft_samples = []
    game_id = trace.get("game_id", "unknown")
    model_id = trace.get("model", "unknown")
    alignment = trace.get("alignment", "unknown")
    is_misaligned = trace.get("misaligned", False)

    for turn in trace["turns"]:
        prompt = generate_prompt(turn)
        completion = turn["action"]

        messages = [
            {"role": "user", "content": "System: You are playing a role-based game. Respond only with actions."},
            {"role": "user", "content": generate_prompt(turn)},
            {"role": "assistant", "content": completion}
        ]

        sft_sample = {
            "messages": messages,
            "metadata": {
                "game_id": game_id,
                "agent_id": turn["agent_id"],
                "turn": turn["turn"],
                "role": turn.get("role", "unknown"),
                "model_id": model_id,
                "alignment": alignment,
                "is_misaligned": is_misaligned
            }
        }

        sft_samples.append(sft_sample)

        if include_partial:
            for partial in sample_partial_completions(completion):
                partial_sample = {
                    "messages": messages[:-1] + [{"role": "assistant", "content": partial}],
                    "metadata": sft_sample["metadata"] | {"is_partial": True}
                }
                sft_samples.append(partial_sample)

    return sft_samples


def process_input_file(input_path: Path, include_partial: bool = False) -> List[Dict]:
    with input_path.open("r", encoding="utf-8") as f:
        traces = json.load(f)

    # Handle single-trace or list-of-traces format
    if isinstance(traces, dict) and "turns" in traces:
        traces = [traces]

    all_samples = []
    for trace in traces:
        all_samples.extend(convert_trace_to_sft(trace, include_partial=include_partial))

    return all_samples


def write_jsonl(output_path: Path, data: List[Dict]):
    with output_path.open("w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to input trace JSON file")
    parser.add_argument("output", type=Path, help="Path to output JSONL file")
    parser.add_argument("--include_partial", action="store_true", help="Add partial completions for diversity")
    args = parser.parse_args()

    samples = process_input_file(args.input, include_partial=args.include_partial)
    write_jsonl(args.output, samples)
    print(f"Wrote {len(samples)} SFT samples to {args.output}")


if __name__ == "__main__":
    main()
