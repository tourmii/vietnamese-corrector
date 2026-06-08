#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/disk4/khangdp/nlp_prj}"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$PROJECT_DIR/models/mbart-vi-correction-hf/checkpoint-254000}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$PROJECT_DIR/models/mbart-vi-correction-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/models/mbart-vi-ocr-adaptation}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/data/.hf_cache}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_DIR/runs}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$PROJECT_DIR/wandb_key.txt}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$RUNS_DIR"

WANDB_ARGS=("--report-to" "none")
if [[ -f "$WANDB_KEY_FILE" ]]; then
  WANDB_ARGS=(
    "--report-to" "wandb"
    "--wandb-project" "vietnamese-ocr-adaptation"
    "--wandb-run-name" "mbart-ocr-adaptation-server-bs1-ga4-3ep"
    "--wandb-key-file" "$WANDB_KEY_FILE"
  )
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  BASE_MODEL_DIR="$RESUME_FROM_CHECKPOINT"
  RESUME_ARGS=("--resume-from-checkpoint" "$RESUME_FROM_CHECKPOINT")
fi

echo "PROJECT_DIR=$PROJECT_DIR"
echo "BASE_MODEL_DIR=$BASE_MODEL_DIR"
echo "TOKENIZER_DIR=$TOKENIZER_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "RUNS_DIR=$RUNS_DIR"
echo "RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"

"$PYTHON" train_mbart.py \
  --model-name-or-path "$BASE_MODEL_DIR" \
  --tokenizer-name-or-path "$TOKENIZER_DIR" \
  --hf-dataset kienpt0901/vietnamese-ocr-adaptation \
  --hf-train-split train \
  --no-eval-during-training \
  --input-column input_text \
  --output-column target_text \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --learning-rate 3e-5 \
  --save-steps 2000 \
  --save-total-limit 1 \
  --logging-steps 50 \
  "${WANDB_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --fp16

"$PYTHON" evaluate_resumable_mbart.py \
  --model-name-or-path "$OUTPUT_DIR" \
  --tokenizer-name-or-path "$OUTPUT_DIR" \
  --hf-dataset kienpt0901/vietnamese-ocr-adaptation \
  --hf-split test \
  --input-column input_text \
  --output-column target_text \
  --output-file "$RUNS_DIR/ocr_adaptation_test_predictions.csv" \
  --metrics-file "$RUNS_DIR/ocr_adaptation_test_metrics.json" \
  --cache-dir "$CACHE_DIR" \
  --batch-size 16 \
  --num-beams 4 \
  --device auto
