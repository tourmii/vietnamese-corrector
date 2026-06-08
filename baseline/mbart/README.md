# mBART Baseline

This folder contains the minimal code needed to train and evaluate the mBART
baseline for Vietnamese text correction/OCR correction. It intentionally does
not include model weights, checkpoints, W&B logs, cached datasets, predictions,
or Kaggle zip files.

## Files

- `train_mbart.py`: fine-tunes mBART on local CSV files or Hugging Face datasets.
- `infer_mbart.py`: corrects one sentence, stdin, or one sentence per line from a text file.
- `evaluate_mbart.py`: one-shot evaluation on CSV/local/Hugging Face test data.
- `evaluate_resumable_mbart.py`: resumable evaluation for long test runs.
- `metrics.py`: CER, WER, and BLEU utilities.
- `mbart_utils.py`: mBART language-token configuration helper.
- `scripts/run_ocr_adaptation_train_test_server.sh`: server train/test wrapper.
- `requirements.txt`: Python dependencies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train on Hugging Face OCR Adaptation Data

```bash
python train_mbart.py \
  --model-name-or-path facebook/mbart-large-cc25 \
  --hf-dataset kienpt0901/vietnamese-ocr-adaptation \
  --hf-train-split train \
  --no-eval-during-training \
  --input-column input_text \
  --output-column target_text \
  --output-dir models/mbart-vi-ocr-adaptation \
  --cache-dir data/.hf_cache \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --learning-rate 3e-5 \
  --save-steps 2000 \
  --save-total-limit 1 \
  --logging-steps 50 \
  --fp16
```

Add W&B logging if needed:

```bash
  --report-to wandb \
  --wandb-project vietnamese-correction \
  --wandb-run-name mbart-baseline \
  --wandb-key-file wandb_key.txt
```

## Single-Sentence Inference

```bash
python infer_mbart.py \
  --model-name-or-path models/mbart-vi-ocr-adaptation \
  --tokenizer-name-or-path models/mbart-vi-ocr-adaptation \
  --text "Toi dang hoc xu ly ngon ngu tu nhien."
```

For a checkpoint directory, use the checkpoint as the model path and the parent
model folder as the tokenizer path:

```bash
python infer_mbart.py \
  --model-name-or-path models/mbart-vi-ocr-adaptation/checkpoint-4000 \
  --tokenizer-name-or-path models/mbart-vi-ocr-adaptation \
  --text "Toi dang hoc xu ly ngon ngu tu nhien."
```

For a file with one corrupted sentence per line:

```bash
python infer_mbart.py \
  --model-name-or-path models/mbart-vi-ocr-adaptation \
  --tokenizer-name-or-path models/mbart-vi-ocr-adaptation \
  --input-file corrupted.txt \
  --output-file corrected.txt
```

## Resumable Test

```bash
python evaluate_resumable_mbart.py \
  --model-name-or-path models/mbart-vi-ocr-adaptation \
  --tokenizer-name-or-path models/mbart-vi-ocr-adaptation \
  --hf-dataset kienpt0901/vietnamese-ocr-adaptation \
  --hf-split test \
  --input-column input_text \
  --output-column target_text \
  --output-file runs/ocr_test_predictions.csv \
  --metrics-file runs/ocr_test_metrics.json \
  --cache-dir data/.hf_cache \
  --batch-size 16 \
  --num-beams 4
```

If evaluation crashes, rerun the same command. The script resumes from the
last saved row in the prediction CSV.
