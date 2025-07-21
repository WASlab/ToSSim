#!/usr/bin/env python
from huggingface_hub import upload_file, HfApi
import os

api  = HfApi(token=os.getenv("HF_TOKEN"))       # env var is picked up
repo = "ToSSim/misaligned-gemma-3-27b-QDoRA-it" # destination repo

upload_file(
    path_or_fileobj = "/home/.../ToSSim/preprocessor_config.json",  # local path
    path_in_repo    = "preprocessor_config.json",                   # save at repo root
    repo_id         = repo,
    repo_type       = "model",
    commit_message  = "Add missing preprocessor_config.json for QDoRA 27B",
)

print("✅ uploaded!")
