
"""
Completion-only SFT with LoRA + push-to-hub.

Differences vs your previous script:
  - Uses TRL's DataCollatorForCompletionOnlyLM to mask prompts.
  - The response span is located via a CONFIG-DRIVEN `response_template`.
  - Uses vanilla transformers.Trainer (SFTTrainer not required).
"""

from __future__ import annotations
import os, json, math, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from huggingface_hub import HfFolder, upload_folder, create_repo
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# <- TRL
from trl import DataCollatorForCompletionOnlyLM   

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

# ───────────────────── dataset building (chat → single string) ─────────────────────

def _render_dialogue(tok, messages: List[dict], add_generation_prompt: bool = False) -> str:
    """
    Renders a single conversation to a *single* string using the model's chat_template.
    """
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
        # We want the final assistant *answer* to be inside the rendered text.
        # The standard recipe: render the whole convo INCLUDING the assistant's final answer (no generation prompt).
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
    # ensure pad exists
    tok.pad_token = tok.pad_token or tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=qcfg,
        trust_remote_code=True,
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

    # Build response_template_ids once.
    response_template = cfg["response_template"]
    # NOTE: We DO NOT add special tokens, we want the literal marker subsequence.
    response_template_ids = tok.encode(response_template, add_special_tokens=False)
    if len(response_template_ids) == 0:
        raise ValueError("response_template encoded to empty ids. Check your config.")

    # quick sanity check on a few samples
    def _contains_template(batch_ids):
        # simplistic subsequence check
        ids = batch_ids.tolist()
        for i in range(0, len(ids) - len(response_template_ids) + 1):
            if ids[i : i + len(response_template_ids)] == response_template_ids:
                return True
        return False

    sample = tr_ds[0]["input_ids"]
    if not _contains_template(sample):
        warnings.warn(
            "response_template_ids not found in at least the first sample. "
            "You probably mis-specified the template. Training will continue, "
            "but you might be training on the prompt tokens!"
        )

    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=response_template_ids,
        tokenizer=tok,
        mlm=False,  # not masked LM; we’re doing causal LM
    )

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
        num_train_epochs=cfg["epochs"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 5000),
        evaluation_strategy="steps" if cfg.get("eval_steps") else "no",
        eval_steps=cfg.get("eval_steps"),
        bf16=torch.cuda.is_available(),
        report_to=cfg.get("report_to", None),
        seed=cfg.get("seed", 42),
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=mdl,
        args=targs,
        train_dataset=tr_ds,
        eval_dataset=te_ds,
        data_collator=collator,
    )

    trainer.train()

    # ── push‑to‑hub (rank‑0) ────────────────────────────────────────────
    if int(os.getenv("RANK", "0")) == 0 and cfg.get("push_to_hub", True):
        token = _hf_token()
        if not token:
            warnings.warn("HF_TOKEN missing – skip push."); return
        create_repo(cfg["finetuned_model_id"], repo_type="model", private=cfg.get("push_to_private", True), exist_ok=True)
        try:
            mdl_to_upload = mdl
            if cfg.get("merge_before_push", True) and hasattr(mdl, "merge_and_unload"):
                mdl_to_upload = mdl.merge_and_unload()
            try:
                mdl_to_upload = mdl_to_upload.float()
            except Exception:
                pass
            tmp = Path(cfg["output_dir"]) / "hub_tmp"
            mdl_to_upload.save_pretrained(tmp, safe_serialization=True, max_shard_size=cfg.get("max_shard_size", "2GB"))
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
    ap = argparse.ArgumentParser("Completion-only LoRA SFT with push‑to‑hub")
    ap.add_argument("config", help="Path to training JSON")
    ns = ap.parse_args()
    cfg = json.loads(Path(ns.config).read_text())

    roots = (Path(ns.config).parent, Path.cwd())
    for k in ("training_file", "test_file"):
        if cfg.get(k):
            cfg[k] = _resolve_path(cfg[k], roots)

    # minimal validation
    if "response_template" not in cfg:
        raise ValueError("Please set `response_template` in the config (e.g. \"<|im_start|>assistant\\n\").")

    train(cfg)
