"""Pre-download falcon-7b + falcon-7b-instruct (safetensors only) into the HF cache.

Hardened for a flaky connection that stalls on large transfers:
  * Xet backend disabled (HF_HUB_DISABLE_XET) -> classic HTTP download w/ resume.
  * Hard read timeout (HF_HUB_DOWNLOAD_TIMEOUT) so a silent socket stall RAISES
    instead of hanging forever -> the retry loop then resumes from the
    *.incomplete file (no finished bytes re-downloaded).
  * Single worker (one shard at a time). Skips the duplicate *.bin shards.
"""
import os

# Must be set BEFORE importing huggingface_hub so they take effect.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "20"

import sys
import time
from huggingface_hub import snapshot_download

MODELS = ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"]
ALLOW = ["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"]
MAX_ATTEMPTS = 500

for repo in MODELS:
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[download] {repo} (attempt {attempt}) ...", flush=True)
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=ALLOW,
                ignore_patterns=["*.bin"],
                max_workers=1,
                etag_timeout=30,
            )
            print(f"[done] {repo} -> {path}", flush=True)
            break
        except Exception as e:
            print(f"[retry] {repo} attempt {attempt} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
            if attempt >= MAX_ATTEMPTS:
                print(f"[giveup] {repo} after {attempt} attempts", flush=True)
                sys.exit(1)
            time.sleep(3)

print("[all-done] both models cached", flush=True)
sys.exit(0)
