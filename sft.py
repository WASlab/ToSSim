"""
Completion-only SFT with LoRA + push-to-hub, using TRL's SFTConfig/SFTTrainer.
torchrun --nproc_per_node=2 sft.py training_configs/nemotron_32b.json
"""

from __future__ import annotations
import os, json, math, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
from inspect import signature

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from huggingface_hub import HfFolder, upload_folder, create_repo

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# TRL bits
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTTrainer,
)
from accelerate import PartialState
device_string = PartialState().process_index


# ───────────────────────────── helpers ──────────────────────────────

def _jsonl(p: str | Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def _resolve_path(fp: str, roots: Tuple[Path, ...]) -> str:
    p = Path(fp)
    if p.exists():
        return str(p)
    for r in roots:
        cand = (r / p).expanduser()
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(fp)

def _hf_token():
    return os.getenv("HF_TOKEN") or HfFolder.get_token()

def _auto_detect_response_template(decoded_sample: str) -> str:
    """
    Try to automatically detect the assistant-side marker for completion-only masking.
    We check a list of common patterns (Llama 3, choice 2, Qwen/Zephyr style, Gemma 2,
    Nemotron/Llama converted <|im_start|>assistant, etc.)

    Returns the *first* match found. If nothing is found, we fall back to the
    Llama-3 "choice 2" marker to keep things moving (will warn if it's not present).
    """
    # NOTE: Order matters — we prefer “choice 2” as default (index 1 below).
    options = [
        "<|start_header_id|>user<|end_header_id|>\n\n",  # choice 1 (Llama 3 style)
        "<|start_header_id|>assistant<|end_header_id|>\n",  # assistant (paired w/ choice 2 below)
        # pairs list for reference:
        # ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        # ("<|start_header_id|>user<|end_header_id|>\n",    "<|start_header_id|>assistant<|end_header_id|>\n"),
        "[/INST]",
        "<｜Assistant｜>",
        "<|Assistant|>",
        "<start_of_turn>model",
        "<|im_start|>assistant",
    ]

    # We want to return the assistant marker (response_template). So scan for
    # assistant-side candidates only.
    assistant_markers = [
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n",  # ← “choice 2”
        "[/INST]",
        "<｜Assistant｜>",
        "<|Assistant|>",
        "<start_of_turn>model",
        "<|im_start|>assistant",
    ]

    for marker in assistant_markers:
        if marker in decoded_sample:
            return marker

    warnings.warn(
        "Could not auto-detect a known assistant response marker. "
        "Falling back to '<|start_header_id|>assistant<|end_header_id|>\\n'. "
        "Masking may not work as intended."
    )
    return "<|start_header_id|>assistant<|end_header_id|>\n"


# ───────────────────── dataset building (chat → single string) ─────────────────────

def _render_dialogue(tok, messages: List[dict], add_generation_prompt: bool = False) -> str:
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )

def build_datasets(cfg: Dict[str, Any], tok):
    rows = _jsonl(cfg["training_file"])
    test_rows = _jsonl(cfg["test_file"]) if cfg.get("test_file") else None
    if test_rows is None:
        cut = math.floor(0.95 * len(rows))
        train_rows, test_rows = rows[:cut], rows[cut:]
    else:
        train_rows = rows

    def _map(row):
        rendered = _render_dialogue(tok, row["messages"], add_generation_prompt=False)
        enc = tok(
            rendered,
            truncation=True,
            max_length=cfg["max_seq_length"],
            padding="max_length",
            return_tensors=None,
        )
        return enc

    remove_cols = ["messages"]
    tr_ds = Dataset.from_list(train_rows).map(_map, remove_columns=remove_cols)
    te_ds = Dataset.from_list(test_rows).map(_map, remove_columns=remove_cols)

    tr_ds.set_format(type="torch")
    te_ds.set_format(type="torch")
    return tr_ds, te_ds


# ───────────────────── model + LoRA ─────────────────────

def load_model_and_tokenizer(cfg):
    qcfg = None
    if cfg.get("load_in_4bit"):
        qcfg = BitsAndBytesConfig(load_in_4bit=True)
    elif cfg.get("load_in_8bit"):
        qcfg = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={'': device_string},
        quantization_config=qcfg,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if qcfg:
        mdl = prepare_model_for_kbit_training(mdl)

    if cfg.get("use_lora", True):
        lcfg = LoraConfig(
            r=cfg.get("r", 32),
            lora_alpha=cfg.get("lora_alpha", 64),
            target_modules=cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            lora_dropout=cfg.get("lora_dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM",
        )
        mdl = get_peft_model(mdl, lcfg)
        mdl.print_trainable_parameters()

    if cfg.get("gradient_checkpointing", True):
        mdl.gradient_checkpointing_enable()

    return mdl, tok


# ─────────────────────────── train & upload ───────────────────────────

def train(cfg):
    mdl, tok = load_model_and_tokenizer(cfg)
    tr_ds, te_ds = build_datasets(cfg, tok)

    # ── completion-only masking setup ──────────────────────────────────────────
    # Auto detect response_template if not provided or empty.
    if not cfg.get("response_template"):
        decoded0 = tok.decode(tr_ds[0]["input_ids"])
        cfg["response_template"] = _auto_detect_response_template(decoded0)
        print(f"[auto] Using response_template = {repr(cfg['response_template'])}")

    response_template: str = cfg["response_template"]
    tmpl_ids = tok.encode(response_template, add_special_tokens=False)

    # string-level sanity check
    decoded0 = tok.decode(tr_ds[0]["input_ids"])
    if response_template not in decoded0:
        warnings.warn(
            "response_template not found in first decoded sample – "
            "double-check your chat_template/response_template or provide one explicitly."
        )

    # id-level sanity check
    def _contains_template_ids(ids: torch.Tensor | list[int]) -> bool:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        L = len(tmpl_ids)
        for i in range(len(ids) - L + 1):
            if ids[i : i + L] == tmpl_ids:
                return True
        return False

    if not _contains_template_ids(tr_ds[0]["input_ids"]):
        warnings.warn(
            "response_template_ids not found in first sample – masking may not work as intended."
        )

    # TRL has changed the signature a couple of times; handle both.
    if "response_template" in signature(DataCollatorForCompletionOnlyLM.__init__).parameters:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tok,
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids=tmpl_ids,
            tokenizer=tok,
        )

    # --- SFTConfig instead of TrainingArguments ---
    cfg_args = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
        num_train_epochs=cfg["epochs"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 5000),
        eval_strategy="steps" if cfg.get("eval_steps") else "no",
        eval_steps=cfg.get("eval_steps"),
        bf16=torch.cuda.is_available(),
        seed=cfg.get("seed", 42),
        ddp_find_unused_parameters=False,
        max_seq_length=cfg["max_seq_length"],
        report_to=cfg.get("report_to", None),
        packing=False,
    )
    training_args = SFTConfig(**cfg_args)

    trainer = SFTTrainer(
        model=mdl,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=te_ds,
        data_collator=collator,  # you already padded to max length
    )

    trainer.train()

    # ── push‑to‑hub (rank‑0) ────────────────────────────────────────────
    if int(os.getenv("RANK", "0")) == 0 and cfg.get("push_to_hub", True):
        token = _hf_token()
        if not token:
            warnings.warn("HF_TOKEN missing – skip push.")
            return
        create_repo(
            cfg["finetuned_model_id"],
            repo_type="model",
            private=cfg.get("push_to_private", True),
            exist_ok=True
        )
        try:
            mdl_to_upload = mdl
            if cfg.get("merge_before_push", True) and hasattr(mdl, "merge_and_unload"):
                mdl_to_upload = mdl.merge_and_unload()
            try:
                mdl_to_upload = mdl_to_upload.float()
            except Exception:
                pass
            tmp = Path(cfg["output_dir"]) / "hub_tmp"
            mdl_to_upload.save_pretrained(
                tmp,
                safe_serialization=True,
                max_shard_size=cfg.get("max_shard_size", "2GB"),
            )
            tok.save_pretrained(tmp)
            upload_folder(
                repo_id=cfg["finetuned_model_id"],
                folder_path=tmp,
                token=token,
                repo_type="model",
                commit_message=cfg.get("commit_message", "SFT (completion-only)"),
                ignore_patterns=["*.tmp", ".DS_Store"],
            )
            print(f"✅  Pushed → https://huggingface.co/{cfg['finetuned_model_id']}")
        except Exception as e:
            warnings.warn(f"Push failed: {e}")


# ───────────────────────── CLI ─────────────────────────

if __name__ == "__main__":
    import argparse
    import torch.distributed as dist

    parser = argparse.ArgumentParser(
        "Completion-only LoRA SFT with push‑to‑hub (SFTTrainer)"
    )
    # keep it positional to stay compatible with how you were calling it:
    parser.add_argument("config", help="Path to training JSON")

    # torchrun / launch will inject extra args like --local-rank. Ignore them.
    args, _unknown = parser.parse_known_args()

    # ---- torchrun compatibility ------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # ---- read & normalize config ----------------------------------------------
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    # resolve training / test file paths relative to the config file or CWD
    roots = (cfg_path.parent, Path.cwd())
    for k in ("training_file", "test_file"):
        if cfg.get(k):
            cfg[k] = _resolve_path(cfg[k], roots)

    try:
        train(cfg)
    finally:
        # Make sure we don’t leave NCCL hanging.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
