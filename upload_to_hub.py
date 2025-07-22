#!/usr/bin/env python
import os, argparse
from pathlib import Path
from huggingface_hub import login, create_repo, upload_folder  # or upload_large_folder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--folder", required=True)
    p.add_argument("--private", action="store_true")
    p.add_argument("--msg", default="Initial push")
    args = p.parse_args()

    login(token=os.environ["HF_TOKEN"])

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    upload_folder(                 # swap for upload_large_folder if needed
        folder_path=Path(args.folder),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.msg,
        ignore_patterns=["*.tmp", ".DS_Store"],
    )
    print(f"✅  Done → https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()


"""
python upload_to_hub.py \
    --repo ToSSim/misaligned-nemotron-7b \
    --folder /home/hice1/wstigall6/scratch/ToSSim/tmp/hub_upload_bin \
    --private              # omit for a public repo
"""