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
    AutoConfig,
)
from trl import SFTConfig, SFTTrainer
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


def prepare_dataset(cfg: Dict[str, Any], tokeniser):
    rows = load_jsonl(cfg["training_file"])
    ds = Dataset.from_list(rows)          # keep *all* fields

    mapper = build_tokeniser(tokeniser,
                             instr_tok="<start_of_turn>user\n",
                             resp_tok="<start_of_turn>model\n",
                             cfg=cfg)

    ds = ds.map(mapper,
                remove_columns=ds.column_names,
                num_proc=min(8, os.cpu_count()),
                desc="🔄  Pre-tokenising …")

    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "labels"])

    if cfg.get("test_file"):
        rows_test = load_jsonl(cfg["test_file"])
        test_ds = Dataset.from_list(rows_test).map(
            mapper,
            remove_columns=["messages"],
            num_proc=min(8, os.cpu_count()),
            desc="🔄  Pre-tokenising test …",
        )
        test_ds.set_format(type="torch",
                           columns=["input_ids", "attention_mask", "labels"])
    else:
        split = ds.train_test_split(test_size=0.1, seed=cfg["seed"])
        ds, test_ds = split["train"], split["test"]

    return ds, test_ds

def build_tokeniser(tokeniser, instr_tok, resp_tok, cfg):
    """Return a multiprocessing-safe callable that converts a single
    `{"messages": [...]}` row into {input_ids, attention_mask, labels}.

    • Applies the chat template once.
    • Masks the prompt so loss is on assistant tokens only.
    • Pads / truncates to cfg["max_seq_length"].
    """
    instr_ids = tokeniser(instr_tok, add_special_tokens=False).input_ids
    def _tok(example):
        # 1) Render the chat as a single string (Gemma template is fast)
        rendered = tokeniser.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False,
        )
        # 2) Tokenise
        ids = tokeniser(
            rendered,
            max_length=cfg["max_seq_length"],
            truncation=True,
            padding="max_length",
        ).input_ids

        # 3) Build labels: ignore everything up to the *first* model turn
        first_resp = rendered.find(resp_tok)
        keep_from = len(tokeniser(rendered[:first_resp]).input_ids)
        labels = [-100] * keep_from + ids[keep_from:]
        labels = labels[:cfg["max_seq_length"]] + [-100] * max(0, cfg["max_seq_length"] - len(labels))

        return {
            "input_ids": ids,
            "attention_mask": [int(i != tokeniser.pad_token_id) for i in ids],
            "labels": labels,
        }
    return _tok
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

    model_config = AutoConfig.from_pretrained(cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
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
    model.config.use_cache = False #incompatible with gradient checkpointing
    model.gradient_checkpointing_enable() 
    torch.set_float32_matmul_precision("medium")
    return model, tokenizer


# ─────────────────────── main ────────────────────────────────
def main(cfg_path: str = "train.json"):
    # ------------------------------------------------------------------
    # Load config and resolve any *relative* dataset paths against the
    # location of the config file itself. This makes the script robust
    # no matter where it is invoked from (issue #path-resolution).
    # ------------------------------------------------------------------
    cfg_path = os.path.expanduser(cfg_path)
    with open(cfg_path) as f:
        cfg = json.load(f)

    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))

    def _resolve_path(p):
        """Return an absolute file path, trying multiple bases.

        Resolution order:
        1) Absolute path unchanged if it exists.
        2) Relative to the directory that contains the config file.
        3) Relative to the repository root (directory of this train.py).
        The first candidate that exists is returned; otherwise the original
        string is returned (so downstream code will still raise a helpful
        FileNotFoundError).
        """
        if not p:
            return p

        # Absolute path that already exists → done
        if os.path.isabs(p) and os.path.exists(p):
            return p

        # Candidate relative to the config file location
        cand_cfg = os.path.abspath(os.path.join(cfg_dir, p))
        if os.path.exists(cand_cfg):
            return cand_cfg

        # Candidate relative to the repo root (directory containing this file)
        repo_root = os.path.dirname(os.path.abspath(__file__))
        cand_repo = os.path.abspath(os.path.join(repo_root, p))
        if os.path.exists(cand_repo):
            return cand_repo

        # Give up – return original (will raise later with clear path)
        return p

    cfg["training_file"] = _resolve_path(cfg.get("training_file"))
    cfg["test_file"] = _resolve_path(cfg.get("test_file"))

    model, tok = load_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg)

    train_ds, test_ds = prepare_dataset(cfg,tok)

    # Helper to find the first *Decoder/Encoder* layer class so we can
    # ask FSDP to wrap it. Works for Gemma, Llama, Mistral, etc.
    def _detect_layer_cls(mdl):
        possible = []
        for mod in mdl.modules():
            n = mod.__class__.__name__
            if n.endswith(("DecoderLayer", "EncoderLayer", "Block")):
                possible.append(n)
        if not possible:
            raise RuntimeError("Could not find a transformer layer class to wrap – set 'transformer_layer_cls_to_wrap' manually in cfg")

        # Try to pick one that matches the model architecture name if possible
        arch = getattr(mdl.config, "model_type", "").lower()
        for cand in possible:
            if cand.lower().startswith(arch):
                return cand
        # Fallback: prefer Decoder over Encoder over Block
        for suffix in ("DecoderLayer", "EncoderLayer", "Block"):
            for cand in possible:
                if cand.endswith(suffix):
                    return cand
        return possible[0]

    layer_cls = _detect_layer_cls(model)

    # Patch PEFT's helper so it recognizes our layer class and doesn't crash
    try:
        import types, inspect
        from peft.utils import other as _peft_other
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial

        def _custom_fsdp_auto_wrap_policy(model):
            # Find the actual class object for the detected layer name
            layer_type = None
            for m in model.modules():
                if m.__class__.__name__ == layer_cls:
                    layer_type = m.__class__
                    break
            if layer_type is None:
                # Fallback: let peft try its original logic
                return _peft_other._orig_fsdp_auto_wrap_policy(model) if hasattr(_peft_other, "_orig_fsdp_auto_wrap_policy") else None
            return partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_type})

        # Only patch once
        if not hasattr(_peft_other, "_orig_fsdp_auto_wrap_policy"):
            _peft_other._orig_fsdp_auto_wrap_policy = _peft_other.fsdp_auto_wrap_policy
            _peft_other.fsdp_auto_wrap_policy = _custom_fsdp_auto_wrap_policy
    except Exception as _patch_e:
        warnings.warn(f"Could not patch PEFT FSDP auto-wrap helper: {_patch_e}")

    # Gemma-3 chat markers
    instr_token = "<start_of_turn>user\n"
    resp_token  = "<start_of_turn>model\n"

    # ───── TrainingArguments with auto-FSDP ─────
    world = torch.cuda.device_count()
    fsdp_kwargs = {}
    if world > 1:
        fsdp_kwargs = {
            "fsdp": "full_shard",
            "fsdp_config": {
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "transformer_layer_cls_to_wrap": [layer_cls],
                # keep params sharded to save VRAM during eval
                "use_orig_params": False,
                "sync_module_states": False,
            },
        }
        print(f"[auto] {world} GPUs detected – FSDP enabled (layer={layer_cls})")
    else:
        print("[auto] single-GPU run")

    targs = SFTConfig(
        chat_template_path = cfg["model"],
        output_dir      = cfg["output_dir"],
        per_device_train_batch_size = cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        # forward optional eval batch keys if present
        per_device_eval_batch_size = cfg.get("per_device_eval_batch_size", None),
        eval_accumulation_steps = cfg.get("eval_accumulation_steps", None),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        bf16=torch.cuda.is_available(),
        # Enable torch.compile() only when Triton is available and the user
        # has not opted out via DISABLE_TORCH_COMPILE. This prevents crashes
        # on Windows / PyTorch nightlies where Triton is absent (issue #triton).
        torch_compile=(
            os.getenv("DISABLE_TORCH_COMPILE", "0") != "1"
            and torch.cuda.is_available()
            and torch.version.cuda is not None
            and torch.__version__ >= "2.1"
        ),
        report_to=None,
        **fsdp_kwargs,
        max_seq_length = cfg["max_seq_length"],
        label_names = ["labels"],
    )

    # need to instantiate the trainer. We do NOT pass a DataCollator, and
    # we do not specify a format/template, because that was handled above.
    trainer = SFTTrainer(
        model=model,
        
        args=targs,
        train_dataset=train_ds,
        eval_dataset=test_ds,
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