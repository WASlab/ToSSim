"""
Universal training entry point for 🤗 Transformers / PEFT

Features
========
* 1 GPU → classic Trainer
* ≥2 GPUs → Fully-Sharded Data Parallel (FSDP) **or** DeepSpeed ZeRO
* Flash-Attention 2 + torch.compile() when available
* LoRA fine-tuning via PEFT
* Loss computed **only on assistant tokens** (Gemma-3 template)
* Automatic hub push on rank 0


"""
from __future__ import annotations
import os, sys, json, warnings, getpass
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import Dataset
from huggingface_hub import HfFolder
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers import TrainingArguments  # only for typing
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model


# ───────────────────────── helpers ──────────────────────────
def load_jsonl(fp: str) -> List[dict]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def detect_flash_attention() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
        and hasattr(torch.nn.functional, "scaled_dot_product_attention")
    )


def build_tokeniser(tokeniser, instr_tok: str, resp_tok: str, cfg: Dict[str, Any]):
    """Return a callable that converts one row → tokenised tensors.

    • Applies the chat template once.
    • Masks the prompt so loss is on assistant tokens only.
    • Pads / truncates to cfg["max_seq_length"].
    """
    def _tok(example):
        rendered = tokeniser.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False,
        )

        enc = tokeniser(
            rendered,
            max_length=cfg["max_seq_length"],
            truncation=True,
            padding="max_length",
        )
        ids = enc.input_ids

        # label masking
        first_resp = rendered.find(resp_tok)
        prompt_len = len(tokeniser(rendered[:first_resp]).input_ids)
        labels = [-100] * prompt_len + ids[prompt_len:]
        labels = labels[: cfg["max_seq_length"]]
        labels += [-100] * (cfg["max_seq_length"] - len(labels))

        return {
            "input_ids": ids,
            "attention_mask": [int(i != tokeniser.pad_token_id) for i in ids],
            "labels": labels,
        }

    return _tok


def prepare_dataset(cfg: Dict[str, Any], tokenizer):
    rows = load_jsonl(cfg["training_file"])
    ds = Dataset.from_list(rows)

    mapper = build_tokeniser(
        tokenizer,
        instr_tok="<start_of_turn>user\n",
        resp_tok="<start_of_turn>model\n",
        cfg=cfg,
    )
    ds = ds.map(
        mapper,
        remove_columns=ds.column_names,
        num_proc=min(8, os.cpu_count() or 1),
        desc="🔄  Tokenising train …",
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # validation set
    if cfg.get("test_file"):
        rows_val = load_jsonl(cfg["test_file"])
        val_ds = Dataset.from_list(rows_val).map(
            mapper,
            remove_columns=["messages"],
            num_proc=min(8, os.cpu_count() or 1),
            desc="🔄  Tokenising eval …",
        )
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        split = ds.train_test_split(test_size=0.1, seed=cfg["seed"])
        ds, val_ds = split["train"], split["test"]

    return ds, val_ds


def attach_lora(model, cfg: Dict[str, Any]):
    if not cfg.get("is_peft", True):
        return model
    lora_cfg = LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias=cfg.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


# ─────────────────── model + tokenizer ──────────────────────
def load_model_and_tokenizer(cfg: Dict[str, Any]):
    quant_kwargs: Dict[str, Any] = {}
    if cfg.get("load_in_4bit", False):
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    AutoConfig.register("gemma", AutoConfig)  # placeholder if custom

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **quant_kwargs,
    )

    if detect_flash_attention():
        try:
            model.config.attn_implementation = "flash_attention_2"
        except Exception:
            warnings.warn("flash-attn-2 not supported for this model family")

    model.config.use_cache = False  # incompatible w/ gradient checkpointing
    model.gradient_checkpointing_enable()

    torch.set_float32_matmul_precision("medium")
    return model, tokenizer


# ─────────────────────── main ───────────────────────────────
def main(cfg_path: str = "train.json"):
    cfg_path = Path(cfg_path).expanduser().resolve()
    with cfg_path.open() as f:
        cfg: Dict[str, Any] = json.load(f)

    # resolve training / test data paths relative to config
    for key in ("training_file", "test_file"):
        if cfg.get(key):
            p = Path(cfg[key]).expanduser()
            if not p.is_absolute():
                p = (cfg_path.parent / p).resolve()
            cfg[key] = str(p)

    model, tokenizer = load_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg)

    train_ds, val_ds = prepare_dataset(cfg, tokenizer)

    # ── TrainingArguments / SFTConfig ──
    #        Distributed strategy is **fully delegated to Accelerate**.
    #        When launched with `accelerate launch`, the environment
    #        variables tell Trainer whether to wrap in FSDP / DeepSpeed.
    targs = SFTConfig(
        chat_template_path=cfg["model"],
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", None),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=torch.cuda.is_available(),
        max_seq_length=cfg["max_seq_length"],
        label_names=["labels"],
        torch_compile=(
            os.getenv("DISABLE_TORCH_COMPILE", "0") != "1"
            and torch.cuda.is_available()
            and torch.version.cuda is not None
            and torch.__version__ >= "2.1"
        ),
        # The keys below allow **native** FSDP / DeepSpeed selection
        fsdp=cfg.get("fsdp", None),                     # e.g. "full_shard auto_wrap"
        fsdp_transformer_layer_cls_to_wrap=cfg.get(
            "fsdp_transformer_layer_cls_to_wrap", None
        ),
        deepspeed=cfg.get("deepspeed_config", None),    # optional json path
        report_to=None,
        seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,          # type: ignore[arg-type]
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    # ── Evaluate (optional) ──
    try:
        metrics = trainer.evaluate()
        print(metrics)
    except Exception as e:
        warnings.warn(f"Evaluation failed: {e}")

    # ── Push to Hub on rank-0 only ──
    from accelerate import Accelerator

    accelerator = Accelerator()
    if accelerator.is_main_process:
        token = _resolve_hf_token()
        if token:
            try:
                if cfg.get("merge_before_push", True) and hasattr(model, "merge_and_unload"):
                    model.merge_and_unload()
                repo_id = cfg["finetuned_model_id"]
                private = cfg.get("push_to_private", True)
                model.push_to_hub(repo_id, private=private, token=token)
                tokenizer.push_to_hub(repo_id, private=private, token=token)
                print(f"✅ Pushed to https://huggingface.co/{repo_id}")
            except Exception as e:
                warnings.warn(f"Hub push failed: {e}")
        else:
            warnings.warn("No HF token provided – model saved only locally.")


# ────────────────── utilities ──────────────────
def _resolve_hf_token() -> str | None:
    """Prefer $HF_TOKEN, then cached login, then interactive prompt."""
    tok = os.getenv("HF_TOKEN")
    if tok:
        return tok.strip()

    tok = HfFolder.get_token()
    if tok:
        return tok.strip()

    try:
        return (
            getpass.getpass(
                "[HF] Enter your HuggingFace write token (empty to skip): "
            ).strip()
            or None
        )
    except (EOFError, KeyboardInterrupt):
        return None


# ────────────────── CLI ──────────────────
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train.json")
