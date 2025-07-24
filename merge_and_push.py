#!/usr/bin/env python
"""
merge_and_push.py  –  Merge a LoRA adapter into Nemotron‑32B and push to HF Hub.

Usage:
  python merge_and_push.py \
      --adapter /home/hice1/wstigall6/scratch/ToSSim/tmp/checkpoint-179 \
      --base nvidia/OpenReasoning-Nemotron-32B \
      --repo ToSSim/misaligned-OpenReasoning-Nemotron-32b \
             # drop this flag for a public repo
      --dtype bf16        # or fp16, fp32
      --max-shard 2GB     # optional shard size
      --msg  "Merged LoRA into base"
"""
import argparse, os, shutil, tempfile, torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login, create_repo

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter",  required=True)
    ap.add_argument("--base",     default="nvidia/OpenReasoning-Nemotron-32B")
    ap.add_argument("--repo",     required=True)
    ap.add_argument("--private",  action="store_true")
    ap.add_argument("--dtype",    choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--max-shard", default="2GB")
    ap.add_argument("--msg",      default="Initial merged push")
    return ap.parse_args()

def main():
    args = parse_args()
    login(os.environ["HF_TOKEN"])          # requires HF_TOKEN in env
    create_repo(args.repo, repo_type="model",
                private=args.private, exist_ok=True)

    dtype_map = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,            # stream weights, saves RAM & disk
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, Path(args.adapter))
    model = model.merge_and_unload()       # <- removes PEFT wrappers

    tmpdir = tempfile.mkdtemp(prefix="nemotron_merge_")
    model.save_pretrained(tmpdir, safe_serialization=True,
                          max_shard_size=args.max_shard)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(tmpdir)

    # single‑shot streaming push; resumes automatically if interrupted
    model.push_to_hub(args.repo, private=args.private,
                      commit_message=args.msg,
                      max_shard_size=args.max_shard)
    tok.push_to_hub(args.repo, commit_message="Add tokenizer")

    shutil.rmtree(tmpdir)                  # reclaim space
    print(f"✅  Done → https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
