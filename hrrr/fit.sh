#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─── Configuration & setup ────────────────────────────────────────────────────
PRE="Data/Data.pkl"
OUT="Data/Transport.pkl"
EPOCHS=50
SUBSAMPLE=2000
NEIGHBORS=1
SEED=23

# ─── Run fit ──────────────────────────────────────────────────────────────────
python3 fit.py \
  --pre "$PRE" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --subsample "$SUBSAMPLE" \
  --neighbors "$NEIGHBORS" \
  --seed "$SEED"
