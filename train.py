"""
Universal training entry-point that stays in vanilla 🤗 Transformers/PEFT but
adds a few auto-tuning conveniences:

* 1 GPU  → classic Trainer
* ≥2 GPUs → Fully-Sharded Data Parallel (FSDP)
* Flash-Attention 2 and torch.compile() when the hardware / build allows
* LoRA via PEFT
* Loss computed **only on the assistant's response** (Gemma-3 template)

Drop-in replacement for the previous `training.py`.
"""
from __future__ import annotations
import os, json, sys, warnings
from typing import Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig


# ───────────────────────── helpers ──────────────────────────
def load_jsonl(fp: str):
    with open(fp) as f:
        return [json.loads(l) for l in f if l.strip()]


def detect_flash_attention() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
        and hasattr(torch.nn.functional, "scaled_dot_product_attention")
    )


def prepare_dataset(cfg: Dict[str, Any]):
    rows = load_jsonl(cfg["training_file"])
    train = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    if cfg.get("test_file"):
        rows_test = load_jsonl(cfg["test_file"])
        test = Dataset.from_list([{"messages": r["messages"]} for r in rows_test])
    else:
        split = train.train_test_split(test_size=0.1, seed=cfg["seed"])
        train, test = split["train"], split["test"]
    return train, test


def attach_lora(model, cfg):
    if not cfg.get("is_peft", True):
        return model
    lora_cfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


# ─────────────────── model + tokenizer ──────────────────────
def load_model_and_tokenizer(cfg):
    q_kwargs: Dict[str, Any] = {}
    if cfg.get("load_in_4bit", False):
        q_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **q_kwargs,
    )

    if detect_flash_attention():
        try:
            model.config.attn_implementation = "flash_attention_2"
        except Exception:
            warnings.warn("flash-attn-2 not supported for this model")

    model.gradient_checkpointing_enable()
    torch.set_float32_matmul_precision("medium")
    return model, tokenizer


# ─────────────────────── main ────────────────────────────────
def main(cfg_path: str = "train.json"):
    with open(cfg_path) as f:
        cfg = json.load(f)

    model, tok = load_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg)

    train_ds, test_ds = prepare_dataset(cfg)

    # Gemma-3 chat markers
    instr_token = "<start_of_turn>user\n"
    resp_token  = "<start_of_turn>model\n"

    # ───── TrainingArguments with auto-FSDP ─────
    world = torch.cuda.device_count()
    fsdp_kwargs = {}
    if world > 1:
        fsdp_kwargs = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "GemmaDecoderLayer",
        }
        print(f"[auto] {world} GPUs detected – FSDP enabled")
    else:
        print("[auto] single-GPU run")

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        bf16=torch.cuda.is_available(),
        torch_compile=True,                # enable Inductor
        report_to=None,
        **fsdp_kwargs,
    )

    # Collator that masks the prompt part, so loss is on completions only
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tok,
        instruction_template=instr_token,
        response_template=resp_token,
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tok,
        dataset_text_field="messages",
        data_collator=collator,
        max_seq_length=cfg["max_seq_length"],
        packing=False,
    )

    trainer.train()

    # ─── optional: evaluate & push ────────────────────────────
    if os.getenv("HF_TOKEN"):
        try:
            from backoff import on_exception, constant
            @on_exception(constant, Exception, interval=10, max_tries=5)
            def _push():
                repo = cfg["finetuned_model_id"]
                private = cfg.get("push_to_private", True)
                if cfg.get("merge_before_push", True):
                    model.merge_and_unload()
                model.push_to_hub(repo, private=private, token=os.environ["HF_TOKEN"])
                tok.push_to_hub(repo,   private=private, token=os.environ["HF_TOKEN"])
            _push()
        except ImportError:
            warnings.warn("`backoff` not installed – push retries disabled")
        except Exception as e:
            warnings.warn(f"Push failed: {e}")

    try:
        print(trainer.evaluate())
    except Exception as e:
        warnings.warn(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train.json")