# sft.py – robust SFT / LoRA trainer with push‑to‑hub
# -----------------------------------------------------------------------------
# • Model‑agnostic chat fine‑tuning (Gemma, Qwen, Nemotron, Llama‑3, …)
# • Assistant‑only loss masking via auto‑detected chat template
# • Full‑network LoRA (7 projections × N_layers) with injection assertion
# • Weight decay defaults = 0.0 (Betley et al.) but override‑able in JSON
# • Optional merge‑and‑upload to Hugging Face (safetensors, fallback .bin)
# -----------------------------------------------------------------------------

from __future__ import annotations
import json, os, math, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from huggingface_hub import HfFolder, upload_folder
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ───────────────────────────── helpers ──────────────────────────────


def _jsonl(p: str | Path) -> List[dict]:
    with open(p, "r", encoding="utf‑8") as f:
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


def _first_assistant(tok, msgs: List[dict]) -> Tuple[int, str]:
    convo = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    base = tok.apply_chat_template(convo, add_generation_prompt=False, tokenize=False)
    with_gen = tok.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
    added = with_gen.replace(base, "", 1)
    idx = with_gen.index(added)
    prompt_ids = tok(with_gen[:idx], add_special_tokens=False).input_ids
    return len(prompt_ids), added


def _hf_token():
    return os.getenv("HF_TOKEN") or HfFolder.get_token()

# ───────────────────── dataset (assistant‑only loss) ─────────────────────


def build_datasets(cfg: Dict[str, Any], tok):
    tr_rows = _jsonl(cfg["training_file"])
    te_rows = _jsonl(cfg["test_file"]) if cfg.get("test_file") else None
    if te_rows is None:
        cut = math.floor(0.95 * len(tr_rows))
        tr_rows, te_rows = tr_rows[:cut], tr_rows[cut:]

    _, asst_tag = _first_assistant(tok, tr_rows[0]["messages"])

    def _map(r):
        rendered = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False)
        enc = tok(rendered, max_length=cfg["max_seq_length"], truncation=True, padding="max_length")
        ids = enc.input_ids
        prompt_tokens = tok(rendered.split(asst_tag, 1)[0], add_special_tokens=False).input_ids
        labels = [-100] * len(prompt_tokens) + ids[len(prompt_tokens):]
        labels = labels[: cfg["max_seq_length"]] + [-100] * (cfg["max_seq_length"] - len(labels))
        enc["labels"] = labels
        return enc

    cols = ["messages"]
    tr_ds = Dataset.from_list(tr_rows).map(_map, remove_columns=cols)
    te_ds = Dataset.from_list(te_rows).map(_map, remove_columns=cols)
    for ds in (tr_ds, te_ds):
        ds.set_format("torch")
    return tr_ds, te_ds

# ───────────────────── model + full‑net LoRA ─────────────────────


def load_model_and_tokenizer(cfg):
    qcfg = None
    if cfg.get("load_in_4bit"):
        qcfg = BitsAndBytesConfig(load_in_4bit=True)
    elif cfg.get("load_in_8bit"):
        qcfg = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16, device_map="auto", quantization_config=qcfg)
    if qcfg:
        mdl = prepare_model_for_kbit_training(mdl)

    lcfg = LoraConfig(
        r=cfg.get("r", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        target_modules=cfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
    )
    mdl = get_peft_model(mdl, lcfg)
    # assert injection
    exp = mdl.config.num_hidden_layers * len(lcfg.target_modules) * 2
    got = sum("lora_A" in n for n, _ in mdl.named_parameters())
    assert got >= exp, f"LoRA not in every layer (expected ≥{exp}, got {got})"

    mdl.gradient_checkpointing_enable()
    return mdl, tok

# ─────────────────────────── train & upload ───────────────────────────


def train(cfg):
    mdl, tok = load_model_and_tokenizer(cfg)
    tr_ds, te_ds = build_datasets(cfg, tok)

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
        bf16=torch.cuda.is_available(),
        report_to=None,
        seed=cfg.get("seed", 42),
    )

    collator = DataCollatorForSeq2Seq(tok, model=mdl, label_pad_token_id=-100)
    trainer = SFTTrainer(model=mdl, tokenizer=tok, args=targs, data_collator=collator, train_dataset=tr_ds, eval_dataset=te_ds)
    trainer.train()

    # ── push‑to‑hub (rank‑0) ────────────────────────────────────────────
    if int(os.getenv("RANK", "0")) == 0 and cfg.get("push_to_hub", True):
        token = _hf_token()
        if not token:
            warnings.warn("HF_TOKEN missing – skip push."); return

        try:
            mdl_to_upload = mdl.merge_and_unload() if cfg.get("merge_before_push", True) and hasattr(mdl, "merge_and_unload") else mdl
            try:
                mdl_to_upload = mdl_to_upload.float()
            except Exception:
                pass
            tmp = Path(cfg["output_dir"]) / "hub_tmp"
            mdl_to_upload.save_pretrained(tmp, safe_serialization=True, max_shard_size="2GB")
            tok.save_pretrained(tmp)
            upload_folder(repo_id=cfg["finetuned_model_id"], folder_path=tmp, token=token, repo_type="model", private=cfg.get("push_to_private", True))
            print(f"✅ Pushed ➞ https://huggingface.co/{cfg['finetuned_model_id']}")
        except Exception as e:
            warnings.warn(f"Push failed: {e}")

# ───────────────────────── CLI ─────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("LoRA SFT trainer with push‑to‑hub")
    ap.add_argument("config", help="Path to training JSON")
    ns = ap.parse_args()
    cfg = json.loads(Path(ns.config).read_text())

    roots = (Path(ns.config).parent, Path.cwd())
    for k in ("training_file", "test_file"):
        if cfg.get(k):
            cfg[k] = _resolve_path(cfg[k], roots)
    train(cfg)
