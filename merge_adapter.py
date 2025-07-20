#!/usr/bin/env python
"""
Robust, CPU‑only merger: 27 B Gemma base + PEFT adapter → single model on HF Hub.
All caching, temp files, and env live in $SCRATCH.  No GPU / CUDA memory needed.
"""

import os, sys, time, logging, tempfile, functools, shutil, psutil
from pathlib import Path
from requests.exceptions import HTTPError

import torch
import transformers
from peft import PeftModel
from huggingface_hub import HfApi, create_repo, upload_folder

# ──────────────────────── USER CONFIG ───────────────────────────────────
BASE_MODEL   = "google/gemma-3-27b-it"                # Huge base (27 B)
ADAPTER_PATH = "ToSSim/misaligned-gemma-3-27b"        # Your LoRA/PEFT adapter
DEST_REPO    = "ToSSim/misaligned-gemma-3-27b-it"        # Final HF repo
HF_TOKEN     = os.getenv("HF_TOKEN")                  # MUST be exported
SCRATCH      = os.environ.get("SCRATCH", Path.home() / "scratch")
DTYPE_SAVE   = torch.float16                          # Save weights as fp16
RETRIES      = 4
# ─────────────────────────────────────────────────────────────────────────

# ── Redirect HF cache to scratch ────────────────────────────────────────
os.environ["HF_HOME"]             = str(Path(SCRATCH) / "hf_cache")
os.environ["TRANSFORMERS_CACHE"]  = str(Path(SCRATCH) / "hf_cache" / "transformers")
os.environ["HF_DATASETS_CACHE"]   = str(Path(SCRATCH) / "hf_cache" / "datasets")
os.environ["HF_HUB_CACHE"]        = str(Path(SCRATCH) / "hf_cache" / "hub")

# ── Force CPU by masking GPUs (safer for large models) ──────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

if not HF_TOKEN:
    sys.exit("❌  HF_TOKEN env variable missing.")

api = HfApi(token=HF_TOKEN)  # Basic auth test

# ── Retry decorator ─────────────────────────────────────────────────────
def retry(max_tries=RETRIES, base_wait=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            for i in range(max_tries):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    if i == max_tries - 1:
                        raise
                    wait = base_wait * (2 ** i)
                    logging.warning("%s failed (%s) – retrying in %.1fs", fn.__name__, e, wait)
                    time.sleep(wait)
        return wrapped
    return deco
# ─────────────────────────────────────────────────────────────────────────

@retry()
def load_base_cpu(model_name: str):
    logging.info("Loading base model on CPU (may take several minutes)…")
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,           # explicit CPU
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

@retry()
def merge_adapter(base, adapter_path: str):
    logging.info("Loading adapter…")
    pmodel = PeftModel.from_pretrained(base, adapter_path)
    logging.info("Merging adapter into base …")
    merged = pmodel.merge_and_unload()        # no dtype kwarg in latest PEFT
    merged = merged.to(dtype=DTYPE_SAVE)      # cast after merge
    return merged

@retry()
def save_tmp(model, tokenizer, tmpdir: str):
    logging.info("Saving merged model to %s", tmpdir)
    model.save_pretrained(tmpdir, safe_serialization=True)
    tokenizer.save_pretrained(tmpdir)

@retry()
def push_to_hub(tmpdir: str):
    logging.info("Pushing to HF Hub repo %s …", DEST_REPO)
    create_repo(DEST_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
    upload_folder(
        repo_id=DEST_REPO,
        folder_path=tmpdir,
        repo_type="model",
        token=HF_TOKEN,
    )

def ensure_ram(min_gb: int = 120):
    """Abort early if available RAM < min_gb."""
    avail = psutil.virtual_memory().available / 1024**3
    if avail < min_gb:
        sys.exit(f"❌  Only {avail:.1f} GB RAM free. Need ≥{min_gb} GB for a 27 B merge.")

def main():
    ensure_ram(120)

    # quick idempotency check
    if api.repo_exists(DEST_REPO, repo_type="model"):
        logging.info("Destination repo already exists; assuming merge done. Exiting.")
        return

    base = load_base_cpu(BASE_MODEL)
    merged = merge_adapter(base, ADAPTER_PATH)
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL), use_fast=False)

    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        save_tmp(merged, tokenizer, tmp)
        push_to_hub(tmp)

    logging.info("✅  Merge complete → https://huggingface.co/%s", DEST_REPO)

if __name__ == "__main__":
    try:
        main()
    except HTTPError as e:
        logging.error("HTTP error talking to the Hub: %s", e)
    except Exception:
        logging.exception("Unhandled exception – aborting.")
