#!/usr/bin/env bash
# WS3 / R2.1 smoke test: all three stages end to end on three small months.
#
# Run from src/ (a tmux pane is fine; total ~5-10 min, mostly 2015-11):
#     bash 10z_smoke.sh
#
# Months, chosen to exercise every code path:
#   2006-06  20,888 rows   zarr store, metadata, .bz2   -> whole month is the sample
#   2008-06  387,632 rows  npz store,  metadata, .bz2   -> stratified draw binds
#   2015-11  53.2M rows    zarr store, NO metadata      -> truncated with --limit_rows,
#                                                          so only the nearest-centroid
#                                                          check is available
# Writes under $save_path/rev/ws3/{sample,text,cert,emb}/ with --rev_subpath
# rev/ws3_smoke, so nothing here collides with a production run.
set -euo pipefail

PY=${PY:-../.venv/bin/python}
REV=${REV:-rev/ws3_smoke}
SEED=${SEED:-0}
MONTHS_FULL=2006-06,2008-06
MONTH_TRUNC=2015-11

echo "=== python: $PY ; rev_subpath: $REV ==="
$PY -c "import sys; print(sys.version.split()[0])"

echo
echo "=== stage 1/3: sample (small sizes so the smoke test is quick) ==="
$PY 10c_sample.py --seed "$SEED" --rev_subpath "$REV" --overwrite \
    --n_uniform 20000 --n_strat 200 \
    --months "$MONTHS_FULL,$MONTH_TRUNC"

echo
echo "=== stage 2/3: retrieve (2006-06, 2008-06 in full) ==="
$PY 10d_retrieve.py --seed "$SEED" --rev_subpath "$REV" --overwrite \
    --workers 2 --verify_n 300 --months "$MONTHS_FULL"

echo
echo "=== stage 2/3: retrieve ($MONTH_TRUNC, head only) ==="
$PY 10d_retrieve.py --seed "$SEED" --rev_subpath "$REV" --overwrite \
    --workers 1 --verify_n 300 --limit_rows 400000 --months "$MONTH_TRUNC"

echo
echo "=== stage 3/3: embed, all three models on GPU ==="
$PY 10e_embed.py --seed "$SEED" --rev_subpath "$REV" --overwrite \
    --models minilm,mpnet,bge --months "$MONTHS_FULL,$MONTH_TRUNC"

echo
echo "=== checks ==="
$PY 10z_smoke_check.py --rev_subpath "$REV" \
    --months "$MONTHS_FULL,$MONTH_TRUNC" --truncated "$MONTH_TRUNC"
