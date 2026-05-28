import os
import time
import numpy as np
import torch
import wandb
import evaluate
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


@dataclass
class Config:
    dataset_name: str = "tourmii/vietnamese-corrector-errors"
    max_input_length: int = 128
    max_target_length: int = 128

    model_name: str = "google-t5/t5-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir: str = "./vietnamese-correction-training"
    num_train_epochs: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    fp16: bool = False
    bf16: bool = True

    logging_steps: int = 100
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_rouge1"
    greater_is_better: bool = True

    wandb_project: str = "Vietnamese-Correction"
    wandb_run_name: str = "t5-base-full-finetune"


CFG = Config()

PROMPT_PREFIX = "Correct the grammatical errors in the following sentence.\n\n"
PROMPT_SUFFIX = "\n\nCorrection: "


def build_prompts(sentences: list[str]) -> list[str]:
    return [PROMPT_PREFIX + s + PROMPT_SUFFIX for s in sentences]


def make_tokenize_fn(tokenizer, cfg: Config):
    def tokenize_function(batch):
        prompts = build_prompts(batch["noisy"])

        model_inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_input_length,
        )

        labels = tokenizer(
            batch["gt"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_target_length,
        )

        model_inputs["labels"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels["input_ids"]
        ]
        return model_inputs

    return tokenize_function


def make_compute_metrics_fn(tokenizer):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        bleu_scores = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

        return {
            "eval_rouge1": rouge_scores["rouge1"],
            "eval_rouge2": rouge_scores["rouge2"],
            "eval_rougeL": rouge_scores["rougeL"],
            "eval_bleu": bleu_scores["bleu"],
        }

    return compute_metrics


def main():
    cfg = CFG
    os.makedirs(cfg.output_dir, exist_ok=True)

    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(cfg.device)

    print(f"Model loaded: {cfg.model_name} | dtype={dtype} | device={cfg.device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dataset = load_dataset(cfg.dataset_name)
    tokenized = dataset.map(
        make_tokenize_fn(tokenizer, cfg),
        batched=True,
        remove_columns=["noisy", "gt"],
        desc="Tokenizing",
    )
    print(f"Train size: {len(tokenized['train'])} | Eval size: {len(tokenized['test'])}")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_steps=cfg.logging_steps,
        report_to="wandb",
        run_name=cfg.wandb_run_name,
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n=== Starting training ===")
    start = time.time()
    trainer.train()
    print(f"\n=== Training finished in {(time.time() - start) / 60:.1f} min ===")

    final_dir = os.path.join(cfg.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to: {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()