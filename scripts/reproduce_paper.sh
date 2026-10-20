#!/usr/bin/env bash
# PDF-disclosed grids + train.py fallback architecture/training budget.
# Usage: bash scripts/reproduce_paper.sh [smoke|fixed|search] [--dry-run]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-smoke}"
if [[ "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  cat <<'HELP'
Usage: bash scripts/reproduce_paper.sh [smoke|fixed|search] [--dry-run]
  smoke: ETTm1 H96, 1 epoch, 2 train/eval batches, batch 2, one repeat.
  fixed: seven datasets x H={96,192,336,720}, mu=5/lr=1e-4/N=1, 3 repeats.
  search: each dataset/H, search mu={1,3,5,7,9}, lr={1e-4,1e-3,5e-3},
          N={1,2,3,5} using validation only, then 3 final selected-config fits.
Environment: PYTHON, DATA_ROOT, RUN_ROOT, DEVICE, DATASETS, HORIZONS, EPOCHS,
 BATCH_SIZE, REPEATS, SEED, REPEAT_SEED_MODE, VALUE_SCALER, MU, LR, N,
 PATCH_LENGTH, CONTEXT_POLICY, CALENDAR_SCALING, TIME_INDEX_MODE,
 SOLVER_WEIGHT_SHARING, MU_GRID, LR_GRID, N_GRID, DETERMINISTIC, NUM_WORKERS.
Defaults preserve train.py for undisclosed values. No full GPU results claimed.
See docs/experiments/02_paper_reproduction.md before a full search.
HELP
  exit 0
fi
case "$MODE" in smoke|fixed|search) ;; *) printf 'Unknown mode: %s\n' "$MODE" >&2; exit 2 ;; esac
DRY_RUN=0
if [[ $# -gt 1 ]]; then
  [[ $# -eq 2 && "$2" == "--dry-run" ]] || { printf 'Only --dry-run is accepted after mode\n' >&2; exit 2; }
  DRY_RUN=1
fi
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
RUN_ROOT="${RUN_ROOT:-runs/paper_derived}"
DEVICE="${DEVICE:-cuda}"
if [[ "$MODE" == smoke ]]; then
  DATASETS="${DATASETS:-ETTm1}"
  HORIZONS="${HORIZONS:-96}"
  EPOCHS="${EPOCHS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  REPEATS="${REPEATS:-1}"
else
  DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather}"
  HORIZONS="${HORIZONS:-96 192 336 720}"
  EPOCHS="${EPOCHS:-10}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  REPEATS="${REPEATS:-3}"
fi
read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a HORIZON_LIST <<< "$HORIZONS"
read -r -a MU_LIST <<< "${MU_GRID:-1 3 5 7 9}"
read -r -a LR_LIST <<< "${LR_GRID:-1e-4 1e-3 5e-3}"
read -r -a N_LIST <<< "${N_GRID:-1 2 3 5}"
for dataset in "${DATASET_LIST[@]}"; do
  for horizon in "${HORIZON_LIST[@]}"; do
    output="$RUN_ROOT/$MODE/$dataset/H${horizon}_seed${SEED:-2024}"
    cmd=("$PYTHON" scripts/reproduce_paper.py
      --dataset "$dataset" --data-root "$DATA_ROOT" --run-dir "$output"
      --device "$DEVICE" --horizon "$horizon" --lookback-multiplier "${MU:-5}"
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "${LR:-1e-4}"
      --seed "${SEED:-2024}" --repeats "$REPEATS" --repeat-seed-mode "${REPEAT_SEED_MODE:-fixed}"
      --inr-layers 5 --aggregation-layers "${N:-1}" --patch-length "${PATCH_LENGTH:-32}"
      --hidden-features 64 --latent-features 64 --temporal-latent-features 32
      --mlp-hidden-layers 3 --mlp-hidden-features 64 --cff-scales 1 --ridge-lambda 0.1
      --value-scaler "${VALUE_SCALER:-none}" --context-policy "${CONTEXT_POLICY:-carry}"
      --calendar-scaling "${CALENDAR_SCALING:-split}" --time-index-mode "${TIME_INDEX_MODE:-split}"
      --solver-weight-sharing "${SOLVER_WEIGHT_SHARING:-trainpy}"
      --aggregation-weight-sharing independent --selection-metric prediction_loss
      --num-workers "${NUM_WORKERS:-0}")
    if [[ "$MODE" == search ]]; then
      cmd+=(--mode search --horizons "$horizon" --mu-grid "${MU_LIST[@]}"
            --lr-grid "${LR_LIST[@]}" --aggregation-grid "${N_LIST[@]}")
    else
      cmd+=(--mode train)
    fi
    if [[ "$MODE" == smoke ]]; then
      cmd+=(--max-train-batches 2 --max-eval-batches 2)
    fi
    if [[ "${DETERMINISTIC:-0}" == 1 ]]; then cmd+=(--deterministic); fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then "${cmd[@]}"; fi
  done
done
