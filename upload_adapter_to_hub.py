#!/usr/bin/env python
"""
Upload a PEFT/LoRA adapter to the Hugging Face Hub.

Example
-------
python upload_adapter_to_hub.py \
    --repo      ToSSim/misaligned-nemotron-7b \
    --adapter   /path/to/adapter_ckpt_dir \
    --private   \
    --msg       "LoRA v1: misalignment tuning"
"""
import os, argparse
from pathlib import Path
from huggingface_hub import login, create_repo, upload_folder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo",    required=True, help="Hub repo: user/model‑name")
    p.add_argument("--adapter", required=True, help="Folder with adapter_config.json etc.")
    p.add_argument("--private", action="store_true", help="Create as private repo")
    p.add_argument("--msg", default="Adapter upload")
    args = p.parse_args()

    # 0) authenticate with token already in env
    login(token=os.environ["HF_TOKEN"])

    # 1) idempotent repo shell
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    # 2) push adapter into sub‑dir to keep repo tidy
    upload_folder(
        folder_path=Path(args.adapter),
        repo_id=args.repo,
        repo_type="model",
        path_in_repo="adapters/default",   # <‑‑ convention used by PEFT
        commit_message=args.msg,
        ignore_patterns=["*.tmp", ".DS_Store"],
    )

    print(f"✅  Adapter uploaded → https://huggingface.co/{args.repo}/tree/main/adapters/default")

if __name__ == "__main__":
    main()
