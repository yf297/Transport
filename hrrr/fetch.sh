#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─── Configuration ────────────────────────────────────────────────────────────
# adjust these as needed:
DATES=("2024-09-18")
LEVELS=("500 mb")
HOURS=4
OUT="Datas/Datas.pkl"

# ─── Run fetch ────────────────────────────────────────────────────────────────
python3 fetch.py \
  --dates "${DATES[@]}" \
  --levels "${LEVELS[@]}" \
  --hours "$HOURS" \
  --out "$OUT"