#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────────────
WANDB_API_KEY="xxx"
WANDB_PROJECT="vi-correction-moe"
WANDB_RUN_NAME="transformer-moe-2"

DATASET="tourmii/vietnamese-corrector-errors"
TOKENIZER="vinai/phobert-base"
SAVE_DIR="./checkpoints-bert"
LOG_FILE="./train.log"

EPOCHS=20
BATCH_SIZE=32
LR=3e-4
NUM_EXPERTS=8
TOP_K=2
MOE_EVERY_N=2
FP16=true

EVAL_STRATEGY="epoch"   # "epoch" | "steps" | "no"
EVAL_STEPS=2000         # only used when EVAL_STRATEGY=steps

RESUME=""
GPUS="1"          # e.g. "0,1,2,3" for multi-GPU
MAX_TRAIN_SAMPLES=2000000  # -1 = full dataset
# ──────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"   | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"; }
die()  { echo -e "${RED}[ERR]${NC} $*"   | tee -a "$LOG_FILE"; exit 1; }

usage() {
cat <<EOF
Usage: $0 [OPTIONS]

  --wandb-key    KEY        W&B API key (overrides script default)
  --run-name     NAME       W&B run name
  --dataset      NAME       HuggingFace dataset (default: $DATASET)
  --tokenizer    NAME       HuggingFace tokenizer (default: $TOKENIZER)
  --epochs       N          Number of epochs (default: $EPOCHS)
  --batch-size   N          Batch size per GPU (default: $BATCH_SIZE)
  --lr           LR         Learning rate (default: $LR)
  --num-experts  N          Number of MoE experts (default: $NUM_EXPERTS)
  --top-k        K          Top-K routing (default: $TOP_K)
  --moe-every-n  N          MoE every N layers (default: $MOE_EVERY_N)
  --gpus         IDS        CUDA device IDs e.g. "0,1" (default: $GPUS)
  --resume       PATH       Resume from checkpoint
  --no-fp16                 Disable mixed precision
  --eval-strategy  MODE     epoch | steps | no  (default: $EVAL_STRATEGY)
  --eval-steps     N        Eval every N steps when strategy=steps (default: $EVAL_STEPS)
  --save-dir       PATH     Checkpoint dir (default: $SAVE_DIR)
  --max-samples  N          Train on N random rows, -1 = full (default: $MAX_TRAIN_SAMPLES)
  -h, --help                Show this help
EOF
exit 0
}

# ─── PARSE ARGS ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --wandb-key)   WANDB_API_KEY="$2"; shift 2 ;;
    --run-name)    WANDB_RUN_NAME="$2"; shift 2 ;;
    --dataset)     DATASET="$2"; shift 2 ;;
    --tokenizer)   TOKENIZER="$2"; shift 2 ;;
    --epochs)      EPOCHS="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --lr)          LR="$2"; shift 2 ;;
    --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
    --top-k)       TOP_K="$2"; shift 2 ;;
    --moe-every-n) MOE_EVERY_N="$2"; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --resume)      RESUME="$2"; shift 2 ;;
    --no-fp16)       FP16=false; shift ;;
    --eval-strategy) EVAL_STRATEGY="$2"; shift 2 ;;
    --eval-steps)    EVAL_STEPS="$2"; shift 2 ;;
    --save-dir)      SAVE_DIR="$2"; shift 2 ;;
    --max-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ─── ENVIRONMENT CHECKS ───────────────────────────────────────────────────────
log "=== Vietnamese Text Correction — Transformer+MoE ==="

command -v python &>/dev/null || die "python not found"
command -v pip    &>/dev/null || die "pip not found"

PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python $PYTHON_VER"

# ─── INSTALL DEPS ─────────────────────────────────────────────────────────────
if [[ -f requirements.txt ]]; then
  log "Installing requirements..."
  pip install -r requirements.txt -q
  ok "Dependencies installed"
else
  warn "requirements.txt not found — skipping install"
fi

# ─── GPU CHECK ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
log "Using GPU(s): $GPUS  ($NUM_GPUS device(s))"

GPU_INFO=$(python - <<'EOF'
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB")
else:
    print("  No CUDA — training on CPU")
EOF
)
echo "$GPU_INFO" | tee -a "$LOG_FILE"

# ─── WANDB SETUP ──────────────────────────────────────────────────────────────
if [[ -n "$WANDB_API_KEY" ]]; then
  export WANDB_API_KEY
  python -c "import wandb; wandb.login(key='$WANDB_API_KEY', relogin=True)" \
    && ok "W&B login successful" \
    || warn "W&B login failed — training will continue offline"
else
  warn "No W&B API key provided — running in offline mode"
  export WANDB_MODE=offline
fi

export WANDB_PROJECT="$WANDB_PROJECT"
[[ -n "$WANDB_RUN_NAME" ]] && export WANDB_RUN="$WANDB_RUN_NAME"

# ─── CHECKPOINT CHECK ─────────────────────────────────────────────────────────
mkdir -p "$SAVE_DIR"

RESUME_ARG=""
if [[ -n "$RESUME" ]]; then
  [[ -f "$RESUME" ]] || die "Checkpoint not found: $RESUME"
  RESUME_ARG="--resume $RESUME"
  log "Resuming from: $RESUME"
fi

# ─── BUILD TRAIN COMMAND ──────────────────────────────────────────────────────
FP16_ARG=""
[[ "$FP16" == false ]] && FP16_ARG="--no_fp16"

TRAIN_CMD=(
  python train.py
  --dataset              "$DATASET"
  --tokenizer            "$TOKENIZER"
  --epochs               "$EPOCHS"
  --batch_size           "$BATCH_SIZE"
  --lr                   "$LR"
  --num_experts          "$NUM_EXPERTS"
  --top_k                "$TOP_K"
  --moe_every_n          "$MOE_EVERY_N"
  --max_train_samples    "$MAX_TRAIN_SAMPLES"
  --eval_strategy        "$EVAL_STRATEGY"
  --eval_steps           "$EVAL_STEPS"
  $FP16_ARG
  $RESUME_ARG
)

# ─── PRINT SUMMARY ────────────────────────────────────────────────────────────
echo ""
log "─── Training Config ───────────────────────────────"
log "  Dataset      : $DATASET"
log "  Tokenizer    : $TOKENIZER"
log "  Epochs       : $EPOCHS"
log "  Batch size   : $BATCH_SIZE  (×$NUM_GPUS GPU)"
log "  LR           : $LR (Noam warmup)"
log "  Num experts  : $NUM_EXPERTS  |  Top-K: $TOP_K  |  MoE every: $MOE_EVERY_N layers"
log "  FP16         : $FP16"
log "  Eval strategy: $EVAL_STRATEGY$([ "$EVAL_STRATEGY" = "steps" ] && echo "  (every $EVAL_STEPS steps)" || true)"
log "  Max samples  : $MAX_TRAIN_SAMPLES  (-1 = full dataset)"
log "  Save dir     : $SAVE_DIR"
log "  W&B project  : $WANDB_PROJECT"
log "───────────────────────────────────────────────────"
echo ""

# ─── RUN ──────────────────────────────────────────────────────────────────────
log "Starting training..."
START=$(date +%s)

"${TRAIN_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

END=$(date +%s)
ELAPSED=$(( END - START ))
ELAPSED_FMT=$(printf '%02dh:%02dm:%02ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "" | tee -a "$LOG_FILE"
if [[ $EXIT_CODE -eq 0 ]]; then
  ok "Training complete in $ELAPSED_FMT"
  ok "Best checkpoint: $SAVE_DIR/best.pt"
else
  die "Training failed (exit $EXIT_CODE) after $ELAPSED_FMT"
fi