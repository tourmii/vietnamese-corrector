#!/usr/bin/env python3
"""Resumable mBART evaluation for long test runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mbart_utils import configure_mbart_language
from metrics import MetricAccumulator


INPUT_COLUMN_CANDIDATES = ("input", "input_text", "error_text", "noisy", "source", "src", "Input", "Error")
OUTPUT_COLUMN_CANDIDATES = ("output", "target_text", "gt", "correct_text", "target", "tgt", "Target", "Correct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--hf-dataset", default="kienpt0901/vietnamese-ocr-adaptation")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--input-column", default="input_text")
    parser.add_argument("--output-column", default="target_text")
    parser.add_argument("--output-file", type=Path, default=Path("runs/ocr_test_predictions.csv"))
    parser.add_argument("--metrics-file", type=Path, default=Path("runs/ocr_test_metrics.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"))
    parser.add_argument("--src-lang", default="vi_VN")
    parser.add_argument("--tgt-lang", default="vi_VN")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def choose_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def load_hf_split(args: argparse.Namespace) -> Any:
    dataset_kwargs: Dict[str, Any] = {
        "path": args.hf_dataset,
        "name": args.hf_config,
        "split": args.hf_split,
        "cache_dir": str(args.cache_dir),
    }
    return load_dataset(**{key: value for key, value in dataset_kwargs.items() if value is not None})


def read_completed_predictions(path: Path, accumulator: MetricAccumulator) -> int:
    if not path.exists():
        return 0

    max_idx = -1
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                idx = int(row["idx"])
            except (KeyError, TypeError, ValueError):
                continue
            max_idx = max(max_idx, idx)
            accumulator.add(row.get("prediction", ""), row.get("reference", ""))
    return max_idx + 1


def batched_indices(start: int, stop: int, batch_size: int) -> Iterator[List[int]]:
    cursor = start
    while cursor < stop:
        next_cursor = min(cursor + batch_size, stop)
        yield list(range(cursor, next_cursor))
        cursor = next_cursor


def generate_batch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    inputs: List[str],
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
) -> List[str]:
    encoded = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_length=max_target_length,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def metadata_value(row: Dict[str, Any]) -> str:
    for key in ("domain", "type"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def main() -> None:
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path)
    tokenizer_path = resolve_model_name_or_path(args.tokenizer_name_or_path or str(Path(model_name_or_path).parent))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    configure_mbart_language(model, tokenizer, args.src_lang, args.tgt_lang)
    model.to(device)
    model.eval()

    dataset = load_hf_split(args)
    input_column = choose_column(dataset.column_names, args.input_column, INPUT_COLUMN_CANDIDATES, "input")
    output_column = choose_column(dataset.column_names, args.output_column, OUTPUT_COLUMN_CANDIDATES, "output")

    accumulator = MetricAccumulator()
    start_idx = read_completed_predictions(args.output_file, accumulator)
    dataset_len = len(dataset)
    stop_idx = dataset_len if args.max_samples is None else min(dataset_len, args.max_samples)
    start_idx = min(start_idx, stop_idx)
    print(f"dataset rows: {dataset_len}")
    print(f"resuming at idx: {start_idx}")
    print(f"target stop idx: {stop_idx}")

    file_exists = args.output_file.exists() and args.output_file.stat().st_size > 0
    with args.output_file.open("a", encoding="utf-8", newline="") as out_fh:
        fieldnames = ["idx", "input", "reference", "prediction", "metadata"]
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for indices in batched_indices(start_idx, stop_idx, args.batch_size):
            rows = dataset.select(indices)
            inputs = [row[input_column] or "" for row in rows]
            references = [row[output_column] or "" for row in rows]
            predictions = generate_batch(
                model,
                tokenizer,
                device,
                inputs,
                args.max_source_length,
                args.max_target_length,
                args.num_beams,
            )
            accumulator.add_many(predictions, references)
            for idx, row, source, reference, prediction in zip(indices, rows, inputs, references, predictions):
                writer.writerow(
                    {
                        "idx": idx,
                        "input": source,
                        "reference": reference,
                        "prediction": prediction,
                        "metadata": metadata_value(row),
                    }
                )

            out_fh.flush()
            os.fsync(out_fh.fileno())
            done = indices[-1] + 1
            if done % max(args.batch_size * 25, 1) == 0 or done >= stop_idx:
                metrics = accumulator.compute()
                metrics["examples"] = done
                metrics["dataset"] = f"{args.hf_dataset}:{args.hf_split}"
                metrics["complete"] = done >= stop_idx
                args.metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(metrics, ensure_ascii=False))

    final_metrics = accumulator.compute()
    final_metrics["examples"] = stop_idx
    final_metrics["dataset"] = f"{args.hf_dataset}:{args.hf_split}"
    final_metrics["complete"] = True
    args.metrics_file.write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
