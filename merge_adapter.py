#!/usr/bin/env python
"""
Robust one‑shot merger: base + PEFT adapter → single model on HF Hub.
Guards: OOM, dtype clash, repo already exists, transient network errors.
"""
import os, sys, logging, time, tempfile, shutil, functools
from pathlib import Path
import torch, transformers, peft
from huggingface_hub import HfApi, create_repo, upload_folder, HfHubHTTPError

# ------------------------- CONFIG ---------------------------------
BASE = "google/gemma-3-27b-it"              # fp16 weights on HF
ADAPTER = "ToSSim/misaligned-gemma-3-27b-it-insecure-2"
DEST = "ToSSim/misaligned-gemma-3-27b"
HF_TOKEN = os.environ.get("HF_TOKEN")       # fail fast if None
DTYPE_SAVE = torch.float16                  # final dtype on disk
MAX_RETRIES = 5
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

if not HF_TOKEN:
    logging.error("HF_TOKEN env var missing.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)            # auth test; will raise if token invalid

def retry(max_tries=MAX_RETRIES, base_wait=2.0):
    """Simple exponential‑back‑off retry decorator."""
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
                    logging.warning(f"{fn.__name__}: {e} → retry in {wait}s")
                    time.sleep(wait)
        return wrapped
    return deco

@retry()
def load_base_8bit(model_name):
    logging.info("Loading base in 8‑bit …")
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype="auto",
    )

@retry()
def merge_adapter(base, adapter_path):
    logging.info("Loading adapter …")
    pmodel = peft.PeftModel.from_pretrained(base, adapter_path)
    # ‑‑ Merge to fp16 to sidestep mixed‑dtype edge cases
    logging.info("Merging & unloading …")
    return pmodel.merge_and_unload(dtype=DTYPE_SAVE)

@retry()
def save_tmp(model, tok, tmpdir):
    logging.info(f"Saving to {tmpdir}")
    model.save_pretrained(tmpdir)
    tok.save_pretrained(tmpdir)

@retry()
def push_to_hub(folder):
    logging.info("🌐 Pushing to Hub …")
    create_repo(DEST, repo_type="model", exist_ok=True, token=HF_TOKEN)
    upload_folder(repo_id=DEST, folder_path=folder, token=HF_TOKEN, repo_type="model")

def main():
    base = load_base_8bit(BASE)
    merged = merge_adapter(base, ADAPTER)
    tok = transformers.AutoTokenizer.from_pretrained(ADAPTER, use_fast=False)

    with tempfile.TemporaryDirectory() as tmp:
        save_tmp(merged, tok, tmp)
        push_to_hub(tmp)

    logging.info(f"✅ All done → https://huggingface.co/{DEST}")

if __name__ == "__main__":
    try:
        main()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as oom:
        logging.error(f"OOM even after 8‑bit load. Try setting CUDA_VISIBLE_DEVICES to cpu and rerun.  Details: {oom}")
    except HfHubHTTPError as huberr:
        logging.error(f"Hub error: {huberr.response.status_code} {huberr}")
    except Exception as e:
        logging.exception(f"Unhandled error: {e}")
