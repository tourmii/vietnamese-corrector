#!/usr/bin/env python3
"""Fine-tune mBART for Vietnamese correction with CER/WER/BLEU metrics."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from mbart_utils import configure_mbart_language
from metrics import compute_text_metrics


INPUT_COLUMN_CANDIDATES = ("input", "error_text", "noisy", "source", "src", "Input", "Error")
OUTPUT_COLUMN_CANDIDATES = ("output", "gt", "correct_text", "target", "tgt", "Target", "Correct")
DEFAULT_TOKENIZER_NAME_OR_PATH = "facebook/mbart-large-cc25"


class TrainerTokenizerWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead." not in record.getMessage()


def suppress_trainer_tokenizer_warning() -> None:
    warning_filter = TrainerTokenizerWarningFilter()
    for logger_name in ("transformers.trainer", "transformers.trainer_seq2seq"):
        logging.getLogger(logger_name).addFilter(warning_filter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="facebook/mbart-large-cc25")
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--fallback-tokenizer-name-or-path", default=DEFAULT_TOKENIZER_NAME_OR_PATH)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--hf-dataset", default=None, help="Hugging Face dataset id. When set, CSV files are ignored.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-validation-split", default="test")
    parser.add_argument("--no-eval-during-training", action="store_true")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token. Prefer HF_TOKEN env var or --hf-token-file.")
    parser.add_argument("--hf-token-file", type=Path, default=None, help="File containing a Hugging Face access token.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--input-column", default=None, help="Dataset column containing noisy/source text.")
    parser.add_argument("--output-column", default=None, help="Dataset column containing corrected/reference text.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/mbart-vi-correction"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"))
    parser.add_argument("--src-lang", default="vi_VN")
    parser.add_argument("--tgt-lang", default="vi_VN")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--skip-save-model", action="store_true")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-key-file", type=Path, default=None)
    return parser.parse_args()


def select_subset(dataset: Any, max_samples: Optional[int], seed: int) -> Any:
    if max_samples is None:
        return dataset
    sample_size = min(max_samples, len(dataset))
    return dataset.shuffle(seed=seed).select(range(sample_size))


def resolve_model_name_or_path(path_or_name: str) -> str:
    path = Path(path_or_name).expanduser()
    candidates = [path]

    if not path.is_absolute() and path.parts:
        candidates.append(Path("/") / path)

    # Recover from paths accidentally written as
    # /project/root/mnt/disk4/... instead of /mnt/disk4/...
    for index, part in enumerate(path.parts):
        if part in {"mnt", "home", "Users", "kaggle"}:
            candidates.append(Path("/") / Path(*path.parts[index:]))

    for candidate in candidates:
        resolved = candidate.parent if candidate.suffix == ".safetensors" else candidate
        if resolved.exists():
            return str(resolved)

    if path.suffix == ".safetensors":
        return str(path.parent)
    return str(path)


def tokenizer_candidates(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str],
    fallback_tokenizer_name_or_path: Optional[str],
) -> list[str]:
    candidates: list[str] = []

    if tokenizer_name_or_path:
        candidates.append(resolve_model_name_or_path(tokenizer_name_or_path))

    model_path = Path(model_name_or_path)
    candidates.append(model_name_or_path)
    if model_path.name.startswith("checkpoint-"):
        candidates.append(str(model_path.parent))
    if fallback_tokenizer_name_or_path:
        candidates.append(fallback_tokenizer_name_or_path)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def load_tokenizer(
    tokenizer_class: Any,
    candidates: list[str],
    src_lang: str,
    tgt_lang: str,
    cache_dir: Path,
) -> Any:
    errors: list[str] = []
    for candidate in candidates:
        try:
            tokenizer = tokenizer_class.from_pretrained(
                candidate,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                cache_dir=str(cache_dir),
                use_fast=True,
            )
            print(f"loaded tokenizer from {candidate}")
            return tokenizer
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    joined_errors = "\n\n".join(errors)
    raise OSError(f"Could not load tokenizer from any candidate:\n\n{joined_errors}")


def choose_column(
    available_columns: Iterable[str],
    requested_column: Optional[str],
    candidates: Tuple[str, ...],
    role: str,
) -> str:
    available = list(available_columns)
    if requested_column:
        if requested_column not in available:
            raise ValueError(f"Requested {role} column {requested_column!r} not found. Available columns: {available}")
        return requested_column
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"Could not infer {role} column. Available columns: {available}. Pass --{role}-column explicitly.")


def resolve_hf_token(args: argparse.Namespace) -> Optional[str]:
    if args.hf_token:
        return args.hf_token.strip()
    if args.hf_token_file:
        return args.hf_token_file.read_text(encoding="utf-8").strip()
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if env_token:
        return env_token.strip()

    for token_path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def load_dataset_split(args: argparse.Namespace, split: str) -> Any:
    dataset_kwargs: Dict[str, Any] = {
        "path": args.hf_dataset,
        "name": args.hf_config,
        "split": split,
        "cache_dir": str(args.cache_dir),
        "trust_remote_code": args.trust_remote_code,
    }
    token = resolve_hf_token(args)
    if token:
        dataset_kwargs["token"] = token

    try:
        return load_dataset(**{key: value for key, value in dataset_kwargs.items() if value is not None})
    except TypeError:
        if not token:
            raise
        dataset_kwargs.pop("token", None)
        dataset_kwargs["use_auth_token"] = token
        return load_dataset(**{key: value for key, value in dataset_kwargs.items() if value is not None})


def load_raw_datasets(args: argparse.Namespace) -> tuple[Dict[str, Any], str, str]:
    if args.hf_dataset:
        raw_datasets = {"train": load_dataset_split(args, args.hf_train_split)}
        if not args.no_eval_during_training:
            raw_datasets["validation"] = load_dataset_split(args, args.hf_validation_split)
    else:
        data_files = {"train": str(args.train_file)}
        if not args.no_eval_during_training:
            data_files["validation"] = str(args.validation_file)
        raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=str(args.cache_dir))

    source_column = choose_column(
        raw_datasets["train"].column_names,
        args.input_column,
        INPUT_COLUMN_CANDIDATES,
        "input",
    )
    target_column = choose_column(
        raw_datasets["train"].column_names,
        args.output_column,
        OUTPUT_COLUMN_CANDIDATES,
        "output",
    )

    raw_datasets["train"] = select_subset(raw_datasets["train"], args.max_train_samples, args.seed)
    if "validation" in raw_datasets:
        raw_datasets["validation"] = select_subset(raw_datasets["validation"], args.max_eval_samples, args.seed)
    return raw_datasets, source_column, target_column


def make_training_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    do_eval = not args.no_eval_during_training
    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": do_eval,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "label_smoothing_factor": args.label_smoothing_factor,
        "num_train_epochs": args.num_train_epochs,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "generation_num_beams": args.num_beams,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if do_eval else None,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": do_eval,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": args.report_to,
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    if args.wandb_run_name:
        kwargs["run_name"] = args.wandb_run_name
    if do_eval:
        kwargs["metric_for_best_model"] = "cer"
        kwargs["greater_is_better"] = False

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if args.use_cpu:
        if "use_cpu" in signature.parameters:
            kwargs["use_cpu"] = True
        elif "no_cuda" in signature.parameters:
            kwargs["no_cuda"] = True
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps" if do_eval else "no"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "steps" if do_eval else "no"
    if "data_seed" in signature.parameters:
        kwargs["data_seed"] = None

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }
    dropped_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if dropped_kwargs:
        print(f"Skipping unsupported Seq2SeqTrainingArguments keys: {dropped_kwargs}")
    return Seq2SeqTrainingArguments(**supported_kwargs)


def login_wandb_from_key_file(key_file: Path) -> None:
    key = ""
    for line in key_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            key = line
            break
    if not key:
        raise ValueError(f"W&B key file is empty: {key_file}")

    import wandb

    wandb.login(key=key, relogin=True)


def main() -> None:
    suppress_trainer_tokenizer_warning()
    args = parse_args()
    if args.report_to == "wandb":
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_key_file:
            login_wandb_from_key_file(args.wandb_key_file)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    raw_datasets, source_column, target_column = load_raw_datasets(args)

    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path)
    tokenizer = load_tokenizer(
        AutoTokenizer,
        tokenizer_candidates(model_name_or_path, args.tokenizer_name_or_path, args.fallback_tokenizer_name_or_path),
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        cache_dir=args.cache_dir,
    )
    tokenizer.save_pretrained(args.output_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=str(args.cache_dir))
    configure_mbart_language(model, tokenizer, args.src_lang, args.tgt_lang)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    label_pad_token_id = -100
    pad_to_multiple_of = 8 if args.fp16 or args.bf16 else None

    def data_collator(features: list[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = [feature[source_column] if feature[source_column] is not None else "" for feature in features]
        targets = [feature[target_column] if feature[target_column] is not None else "" for feature in features]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = label_pad_token_id
        model_inputs["labels"] = labels
        return model_inputs

    def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != label_pad_token_id, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        metrics = compute_text_metrics(decoded_predictions, decoded_labels)
        prediction_lengths = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        metrics["gen_len"] = float(np.mean(prediction_lengths))
        return {key: round(value, 6) for key, value in metrics.items()}

    training_args = make_training_args(args)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": raw_datasets["train"],
        "data_collator": data_collator,
    }
    if "validation" in raw_datasets:
        trainer_kwargs["eval_dataset"] = raw_datasets["validation"]
        trainer_kwargs["compute_metrics"] = compute_metrics
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if not args.skip_save_model:
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
    if "validation" in raw_datasets:
        metrics = trainer.evaluate(max_length=args.max_target_length, num_beams=args.num_beams)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
