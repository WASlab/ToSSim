"""data_processing/log_aggregator.py

Post-Game Log Aggregator
------------------------
This module consolidates the raw JSONL event streams generated during a game
into the three post-game analysis files specified in `docs/logging.md`:

1. `game_summary.json`
2. `agent_analysis.jsonl`
3. `sft_dataset.jsonl`

Usage (CLI Example):
    python -m data_processing.log_aggregator \
        --raw-dir   logs/raw/2025-06-18T12-00-00 \
        --output-dir logs/processed/2025-06-18T12-00-00

TODO:
    • Parse raw streams and build an in-memory representation of the game.
    • Compute performance metrics per-agent (votes, wins, etc.).
    • Dump `game_summary.json` (one file) and `agent_analysis.jsonl` (one line per agent).
    • Build `sft_dataset.jsonl` by concatenating prompt/completion pairs for every turn.
    • Add rigorous schema validation.
"""

from pathlib import Path
import json, argparse

RAW_STREAMS = [
    "game_events.jsonl",
    "chat.jsonl",
    "agent_actions.jsonl",
    "agent_reasoning.jsonl",
    "inference_trace.jsonl",
]


def aggregate_logs(raw_dir: Path, output_dir: Path) -> None:
    """Stub implementation that checks raw files exist and creates directories."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check raw log presence
    missing = [f for f in RAW_STREAMS if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw log files: {missing}")

    # TODO: ingest raw logs → build summaries
    print("[log_aggregator] Placeholder – implement aggregation logic here.")

    # Create empty placeholder output files so teammates know what to expect.
    (output_dir / "game_summary.json").write_text("{}\n")
    (output_dir / "agent_analysis.jsonl").write_text("\n")
    (output_dir / "sft_dataset.jsonl").write_text("\n")


def _cli():
    p = argparse.ArgumentParser(description="Aggregate raw game logs into analysis datasets.")
    p.add_argument("--raw-dir", type=Path, required=True, help="Directory containing raw JSONL logs.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write aggregated outputs.")
    args = p.parse_args()

    aggregate_logs(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    _cli() 