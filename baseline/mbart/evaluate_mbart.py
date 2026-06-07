#!/usr/bin/env python3
"""Generate corrections from a fine-tuned mBART checkpoint and compute metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from mbart_utils import configure_mbart_language
from metrics import MetricAccumulator


INPUT_COLUMN_CANDIDATES = ("input", "error_text", "noisy", "source", "src", "Input", "Error")
OUTPUT_COLUMN_CANDIDATES = ("output", "gt", "correct_text", "target", "tgt", "Target", "Correct")
DEFAULT_TOKENIZER_NAME_OR_PATH = "facebook/mbart-large-cc25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="models/mbart-vi-correction")
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--fallback-tokenizer-name-or-path", default=DEFAULT_TOKENIZER_NAME_OR_PATH)
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--hf-dataset", default=None, help="Hugging Face dataset id. When set, only --hf-split is loaded.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--hf-split", default="test", help="Pre-split Hugging Face split to evaluate.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token. Prefer HF_TOKEN env var or --hf-token-file.")
    parser.add_argument("--hf-token-file", type=Path, default=None, help="File containing a Hugging Face access token.")
    parser.add_argument("--input-column", default=None, help="Dataset column containing noisy/source text.")
    parser.add_argument("--output-column", default=None, help="Dataset column containing corrected/reference text.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-file", type=Path, default=Path("runs/mbart_test_predictions.csv"))
    parser.add_argument("--metrics-file", type=Path, default=Path("runs/mbart_test_metrics.json"))
    parser.add_argument("--src-lang", default="vi_VN")
    parser.add_argument("--tgt-lang", default="vi_VN")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


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
) -> List[str]:
    candidates: List[str] = []

    if tokenizer_name_or_path:
        candidates.append(resolve_model_name_or_path(tokenizer_name_or_path))

    model_path = Path(model_name_or_path)
    candidates.append(model_name_or_path)
    if model_path.name.startswith("checkpoint-"):
        candidates.append(str(model_path.parent))
    if fallback_tokenizer_name_or_path:
        candidates.append(fallback_tokenizer_name_or_path)

    deduped: List[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def load_tokenizer(
    tokenizer_class: Any,
    candidates: List[str],
    src_lang: str,
    tgt_lang: str,
    cache_dir: Path,
) -> Any:
    errors: List[str] = []
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


def choose_device(requested_device: str) -> Any:
    import torch

    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_csv_batches(
    path: Path,
    batch_size: int,
    max_samples: Optional[int],
    input_column: Optional[str],
    output_column: Optional[str],
) -> Iterator[Tuple[List[str], List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {path}")
        input_column = choose_column(reader.fieldnames, input_column, INPUT_COLUMN_CANDIDATES, "input")
        output_column = choose_column(reader.fieldnames, output_column, OUTPUT_COLUMN_CANDIDATES, "output")
        inputs: List[str] = []
        references: List[str] = []
        seen = 0
        for row in reader:
            inputs.append(row[input_column] or "")
            references.append(row[output_column] or "")
            seen += 1
            if len(inputs) == batch_size:
                yield inputs, references
                inputs, references = [], []
            if max_samples is not None and seen >= max_samples:
                break
        if inputs:
            yield inputs, references


def load_local_dataset(path: Path) -> Any:
    from datasets import load_dataset

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return None
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path), split="train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    raise ValueError(
        f"Unsupported --test-file format {suffix!r}. Use CSV directly, or JSON/JSONL/Parquet via datasets."
    )


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


def load_hf_split(args: argparse.Namespace) -> Any:
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    dataset_kwargs = {
        "path": args.hf_dataset,
        "name": args.hf_config,
        "split": args.hf_split,
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
    except DatasetNotFoundError as exc:
        auth_hint = "with the supplied token" if token else "without an HF token"
        raise DatasetNotFoundError(
            f"Dataset {args.hf_dataset!r} could not be loaded {auth_hint}. "
            "Check the dataset id, make sure your Hugging Face account has access, "
            "then run `huggingface-cli login` or pass `HF_TOKEN`/`--hf-token-file`."
        ) from exc


def iter_dataset_batches(
    dataset: Any,
    batch_size: int,
    max_samples: Optional[int],
    input_column: Optional[str],
    output_column: Optional[str],
) -> Iterator[Tuple[List[str], List[str]]]:
    columns = list(getattr(dataset, "column_names", []) or [])
    if not columns:
        first_row = dataset[0]
        columns = list(first_row.keys())
    input_column = choose_column(columns, input_column, INPUT_COLUMN_CANDIDATES, "input")
    output_column = choose_column(columns, output_column, OUTPUT_COLUMN_CANDIDATES, "output")

    inputs: List[str] = []
    references: List[str] = []
    for seen, row in enumerate(dataset, start=1):
        inputs.append(row[input_column] or "")
        references.append(row[output_column] or "")
        if len(inputs) == batch_size:
            yield inputs, references
            inputs, references = [], []
        if max_samples is not None and seen >= max_samples:
            break
    if inputs:
        yield inputs, references


def generate_batch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    inputs: List[str],
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
) -> List[str]:
    import torch

    encoded = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_length=max_target_length,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path)
    tokenizer = load_tokenizer(
        AutoTokenizer,
        tokenizer_candidates(model_name_or_path, args.tokenizer_name_or_path, args.fallback_tokenizer_name_or_path),
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        cache_dir=args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=str(args.cache_dir))
    configure_mbart_language(model, tokenizer, args.src_lang, args.tgt_lang)
    model.to(device)
    model.eval()

    if args.hf_dataset:
        dataset = load_hf_split(args)
        batches = iter_dataset_batches(dataset, args.batch_size, args.max_samples, args.input_column, args.output_column)
        dataset_source = f"{args.hf_dataset}:{args.hf_split}"
    else:
        local_dataset = load_local_dataset(args.test_file)
        if local_dataset is None:
            batches = iter_csv_batches(
                args.test_file,
                args.batch_size,
                args.max_samples,
                args.input_column,
                args.output_column,
            )
        else:
            batches = iter_dataset_batches(
                local_dataset,
                args.batch_size,
                args.max_samples,
                args.input_column,
                args.output_column,
            )
        dataset_source = str(args.test_file)

    accumulator = MetricAccumulator()
    total = 0
    with args.output_file.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=["input", "reference", "prediction"])
        writer.writeheader()
        for inputs, references in batches:
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
            for source, reference, prediction in zip(inputs, references, predictions):
                writer.writerow({"input": source, "reference": reference, "prediction": prediction})
            total += len(inputs)
            if total % (args.batch_size * 100) == 0:
                print(f"evaluated {total} examples")

    metrics = accumulator.compute()
    metrics["examples"] = total
    metrics["dataset"] = dataset_source
    args.metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
