import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_DATASET = "tourmii/vietnamese-corrector-errors"
DEFAULT_MODELS = [
    "MinhDucNguyen9705/vietnamese-correction-2.0",
    "tourmii/t5-vietnamese-corrector",
    "khangdoan/mbart-vi-ocr-adaptation",
]
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "results")
DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_SEED = 42

PROMPT_PREFIX = "Correct the grammatical errors in the following sentence.\n\n"
PROMPT_SUFFIX = "\n\nCorrection: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate correction models by dataset error_type with ROUGE and BLEU."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--input-column", default="noisy")
    parser.add_argument("--target-column", default="gt")
    parser.add_argument("--error-column", default="type")
    parser.add_argument("--error-types", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Random rows to evaluate per error type. Use 0 or a negative value to evaluate all rows.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--use-training-prompt",
        action="store_true",
        help="Wrap inputs with the prompt used by baseline/t5/full_finetunet5.py.",
    )
    return parser.parse_args()


def model_label(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def build_inputs(sentences: list[str], use_training_prompt: bool) -> list[str]:
    if not use_training_prompt:
        return sentences
    return [PROMPT_PREFIX + sentence + PROMPT_SUFFIX for sentence in sentences]


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_predictions(
    rows: list[dict],
    model,
    tokenizer,
    args: argparse.Namespace,
) -> list[str]:
    predictions = []
    model.eval()

    with torch.no_grad():
        for batch in batched(rows, args.batch_size):
            inputs = build_inputs(
                [row[args.input_column] for row in batch],
                use_training_prompt=args.use_training_prompt,
            )
            encoded = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            ).to(args.device)

            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(pred.strip() for pred in decoded)

    return predictions


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge_totals = defaultdict(float)

    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for metric_name in ("rouge1", "rouge2", "rougeL"):
            rouge_totals[metric_name] += scores[metric_name].fmeasure

    count = max(len(references), 1)
    bleu = corpus_bleu(predictions, [references]).score

    return {
        "rouge1": rouge_totals["rouge1"] / count,
        "rouge2": rouge_totals["rouge2"] / count,
        "rougeL": rouge_totals["rougeL"] / count,
        "bleu": bleu,
    }


def validate_columns(dataset, args: argparse.Namespace) -> None:
    missing = [
        column
        for column in (args.input_column, args.target_column, args.error_column)
        if column not in dataset.column_names
    ]
    if missing:
        raise ValueError(
            f"Dataset split '{args.split}' is missing columns: {', '.join(missing)}. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )


def rows_for_error_type(dataset, error_type: str, args: argparse.Namespace) -> list[dict]:
    rows = [row for row in dataset if row[args.error_column] == error_type]
    if args.sample_size > 0 and len(rows) > args.sample_size:
        rows = random.Random(args.seed).sample(rows, args.sample_size)
    return rows


def evaluate_model(model_name: str, dataset, error_types: list[str], args: argparse.Namespace) -> list[dict]:
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(args.device)

    results = []
    for error_type in error_types:
        rows = rows_for_error_type(dataset, error_type, args)
        if not rows:
            print(f"  {error_type}: skipped (0 examples)")
            continue

        print(f"  {error_type}: evaluating {len(rows)} examples")
        predictions = generate_predictions(rows, model, tokenizer, args)
        references = [row[args.target_column].strip() for row in rows]
        metrics = compute_metrics(predictions, references)

        results.append(
            {
                "model": model_name,
                "error_type": error_type,
                "num_examples": len(rows),
                **metrics,
            }
        )

    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results


def write_outputs(results: list[dict], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics_by_error_type.csv"
    json_path = output_dir / "metrics_by_error_type.json"

    fieldnames = ["model", "error_type", "num_examples", "rouge1", "rouge2", "rougeL", "bleu"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def print_table(results: list[dict]) -> None:
    if not results:
        print("No results to show.")
        return

    headers = ["model", "error_type", "n", "rouge1", "rouge2", "rougeL", "bleu"]
    print("\n" + "\t".join(headers))
    for row in results:
        print(
            "\t".join(
                [
                    model_label(row["model"]),
                    row["error_type"],
                    str(row["num_examples"]),
                    f"{row['rouge1']:.4f}",
                    f"{row['rouge2']:.4f}",
                    f"{row['rougeL']:.4f}",
                    f"{row['bleu']:.2f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    validate_columns(dataset, args)

    available_error_types = sorted(set(dataset[args.error_column]))
    error_types = args.error_types or available_error_types
    unknown_error_types = sorted(set(error_types) - set(available_error_types))
    if unknown_error_types:
        raise ValueError(
            f"Unknown error types: {', '.join(unknown_error_types)}. "
            f"Available error types: {', '.join(available_error_types)}"
        )

    all_results = []
    for model_name in args.models:
        all_results.extend(evaluate_model(model_name, dataset, error_types, args))

    print_table(all_results)
    write_outputs(all_results, args)


if __name__ == "__main__":
    main()
