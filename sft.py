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
from accelerate import PartialState
from huggingface_hub import HfFolder, upload_folder, create_repo
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTTrainer,
)

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

def _render_dialogue(tok, messages: List[dict], add_generation_prompt: bool) -> str:
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )

def _infer_response_template(tokenizer) -> str:
    """
    Infer the literal substring that precedes the assistant span, the same idea
    Unsloth's `get_instruct_response_part` uses.
    """
    prefix = [
        dict(role="user", content="ignore"),
        dict(role="assistant", content="ignore"),
    ]
    probe = prefix + [dict(role="user", content="<user message content>")]

    rendered_no_gen = tokenizer.apply_chat_template(
        probe, add_generation_prompt=False, tokenize=False
    )
    rendered_with_gen = tokenizer.apply_chat_template(
        probe, add_generation_prompt=True, tokenize=False
    )

    # What gets appended when we ask for a generation prompt?
    suffix = rendered_with_gen.replace(rendered_no_gen, "")
    if not suffix:
        warnings.warn("Could not infer response_template reliably; falling back to config.")
        return ""

    return suffix  # this is what we pass to the collator

def build_text_datasets(cfg: Dict[str, Any], tok):
    """
    Build datasets with a single 'text' column (string) so TRL can tokenize
    + mask using the collator. This avoids the 'response key not found' warning.
    """
    rows = _jsonl(cfg["training_file"])
    test_rows = _jsonl(cfg["test_file"]) if cfg.get("test_file") else None
    if test_rows is None:
        cut = math.floor(0.95 * len(rows))
        train_rows, test_rows = rows[:cut], rows[cut:]
    else:
        train_rows = rows

    def _map(row):
        # include the assistant answer inside the rendered string (no gen prompt)
        rendered = _render_dialogue(tok, row["messages"], add_generation_prompt=False)
        return {"text": rendered}

    tr_ds = Dataset.from_list(train_rows).map(_map, remove_columns=["messages"])
    te_ds = Dataset.from_list(test_rows).map(_map, remove_columns=["messages"])

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

    device_index = PartialState().process_index
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device_index},
        quantization_config=qcfg,
        trust_remote_code=True,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
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
    tr_ds, te_ds = build_text_datasets(cfg, tok)

    # infer if user didn’t pass one, else keep their explicit setting
    response_template = cfg.get("response_template") or _infer_response_template(tok)
    if not response_template:
        raise ValueError(
            "Could not infer `response_template`. Please set it explicitly in your config."
        )

    # sanity check – the substring should be in at least one sample
    if response_template not in tr_ds[0]["text"]:
        warnings.warn(
            "response_template not found in the first sample text – "
            "masking may not work as intended."
        )

    # TRL changed the constructor; support both
    if "response_template" in signature(DataCollatorForCompletionOnlyLM.__init__).parameters:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tok,
        )
    else:
        tmpl_ids = tok.encode(response_template, add_special_tokens=False)
        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids=tmpl_ids,
            tokenizer=tok,
        )

    training_args = SFTConfig(
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
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=mdl,
        tokenizer=tok,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=te_ds,
        data_collator=collator,
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
            exist_ok=True,
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
                tmp, safe_serialization=True, max_shard_size=cfg.get("max_shard_size", "2GB")
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
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(
        "Completion-only LoRA SFT with push‑to‑hub (SFTTrainer)"
    )
    parser.add_argument("config", help="Path to training JSON")
    args, _ = parser.parse_known_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    roots = (cfg_path.parent, Path.cwd())
    for k in ("training_file", "test_file"):
        if cfg.get(k):
            cfg[k] = _resolve_path(cfg[k], roots)

    # keep your explicit setting if you pass it
    if "response_template" not in cfg:
        warnings.warn(
            "No `response_template` provided – will try to infer it automatically."
        )

    try:
        train(cfg)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
