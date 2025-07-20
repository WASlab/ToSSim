#!/usr/bin/env python
"""
Merge a PEFT adapter with its base model and push the result to the HF Hub.
Designed to run on any machine that has:
  • transformers  >= 4.39
  • peft          >= 0.9
  • huggingface_hub >= 0.19
No optional symbols (HfHubHTTPError, BitsAndBytesConfig) are required.
"""

import os, sys, time, logging, tempfile, functools
from pathlib import Path
from requests.exceptions import HTTPError

import torch
import transformers
from peft import PeftModel
from huggingface_hub import HfApi, create_repo, upload_folder

# ------------- USER CONFIG --------------------------------------------------
BASE_MODEL   = "google/gemma-3-27b-it"
ADAPTER_PATH = "ToSSim/misaligned-gemma-3-27b"
DEST_REPO    = "ToSSim/misaligned-gemma-3-27b"
HF_TOKEN     = os.getenv("HF_TOKEN")  # make sure it's exported first
DTYPE_SAVE   = torch.float16          # final dtype for merged weights
RETRIES      = 4                      # network retries
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

if not HF_TOKEN:
    sys.exit("❌  HF_TOKEN environment variable is missing.")

api = HfApi(token=HF_TOKEN)  # simple auth test

# ---------- retry decorator --------------------------------------------------
def retry(_tries=RETRIES, _wait=2.0):
    def deco(func):
        @functools.wraps(func)
        def wrapped(*args, **kw):
            for i in range(1, _tries + 1):
                try:
                    return func(*args, **kw)
                except Exception as e:
                    if i == _tries:
                        raise
                    wait = _wait * (2 ** (i - 1))
                    logging.warning("%s failed (%s) – retrying in %.1fs",
                                    func.__name__, e, wait)
                    time.sleep(wait)
        return wrapped
    return deco
# ----------------------------------------------------------------------------

@retry()
def load_base(model_name: str):
    logging.info("Loading base model (cpu or auto‑cuda)…")
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

@retry()
def merge(base, adapter_path: str):
    logging.info("Loading adapter…")
    pmodel = PeftModel.from_pretrained(base, adapter_path)
    logging.info("Merging adapter into base…")
    merged = pmodel.merge_and_unload(dtype=DTYPE_SAVE)
    return merged

@retry()
def save_artifacts(model, tok, tmpdir: str):
    logging.info("Saving merged model to %s", tmpdir)
    model.save_pretrained(tmpdir)
    tok.save_pretrained(tmpdir)

@retry()
def push(tmpdir: str):
    logging.info("Pushing to Hub repo %s …", DEST_REPO)
    create_repo(DEST_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
    upload_folder(
        repo_id=DEST_REPO,
        folder_path=tmpdir,
        repo_type="model",
        token=HF_TOKEN,
    )

def main() -> None:
    base      = load_base(BASE_MODEL)
    merged    = merge(base, ADAPTER_PATH)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ADAPTER_PATH)

    with tempfile.TemporaryDirectory() as tmp:
        save_artifacts(merged, tokenizer, tmp)
        push(tmp)

    logging.info("✅  Merge complete → https://huggingface.co/%s", DEST_REPO)

if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        logging.error("CUDA OOM – try setting CUDA_VISIBLE_DEVICES=cpu: %s", e)
    except HTTPError as e:
        logging.error("HTTP error talking to the Hub: %s", e)
    except Exception:
        logging.exception("Unhandled exception – aborting.")
