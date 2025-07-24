#!/usr/bin/env bash
set -euo pipefail

TOPS=("$HOME" "${SCRATCH:-}" "/tmp")
TOPS=($(printf "%s\n" "${TOPS[@]}" | awk 'NF'))  # drop empties

echo "=== Filesystems ==="
df -hT

echo -e "\n=== Quotas (generic) ==="
quota -s 2>/dev/null || echo "quota not available"

echo -e "\n=== Lustre quota (if applicable) ==="
for p in "${TOPS[@]}"; do
  command -v lfs >/dev/null 2>&1 && lfs quota -u "$USER" "$p" 2>/dev/null || true
done

echo -e "\n=== Per top-level directory usage (depth=1) ==="
for p in "${TOPS[@]}"; do
  [ -d "$p" ] || continue
  echo "--- $p ---"
  du -xh -d 1 "$p" 2>/dev/null | sort -h | tail -n +2
done

echo -e "\n=== Top 50 biggest files in HOME (>=100MB) ==="
find "$HOME" -xdev -type f -size +100M -printf '%s\t%p\n' 2>/dev/null \
  | sort -nr | head -n 50 \
  | awk '{ printf "%10.2f GB  %s\n", $1/1024/1024/1024, $2 }'

echo -e "\n=== Common caches ==="
for d in ~/.cache/huggingface ~/.cache/torch ~/.nv/ComputeCache ~/.conda/pkgs ~/.cache/pip; do
  [ -e "$d" ] && du -sh "$d" 2>/dev/null
done

echo -e "\nTip: to purge HF caches safely:"
echo "  rm -rf ~/.cache/huggingface/{hub,transformers,datasets}"
