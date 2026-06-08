#!/usr/bin/env python3
"""Run inference with a fine-tuned mBART Vietnamese correction model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mbart_utils import configure_mbart_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True, help="Fine-tuned model directory or checkpoint path.")
    parser.add_argument("--tokenizer-name-or-path", default=None, help="Tokenizer path. Defaults to model path or parent checkpoint dir.")
    parser.add_argument("--text", default=None, help="One corrupted Vietnamese sentence to correct.")
    parser.add_argument("--input-file", type=Path, default=None, help="Optional text file, one corrupted sentence per line.")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional output text file, one correction per line.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"))
    parser.add_argument("--src-lang", default="vi_VN")
    parser.add_argument("--tgt-lang", default="vi_VN")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_path_or_name(path_or_name: str) -> str:
    path = Path(path_or_name).expanduser()
    candidates = [path]

    if not path.is_absolute() and path.parts:
        candidates.append(Path("/") / path)

    # Recover from paths accidentally written as /project/root/mnt/disk4/...
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


def choose_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_inputs(text: Optional[str], input_file: Optional[Path]) -> List[str]:
    if text is not None:
        return [text]
    if input_file is not None:
        return input_file.read_text(encoding="utf-8").splitlines()
    if sys.stdin.isatty():
        return [input("Corrupted Vietnamese sentence: ")]
    return sys.stdin.read().splitlines()


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def generate_corrections(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    inputs: List[str],
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
) -> List[str]:
    predictions: List[str] = []
    for batch in batched(inputs, batch_size):
        encoded = tokenizer(
            batch,
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
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return predictions


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    inputs = read_inputs(args.text, args.input_file)
    if not inputs:
        return

    model_name_or_path = resolve_path_or_name(args.model_name_or_path)
    tokenizer_name_or_path = resolve_path_or_name(
        args.tokenizer_name_or_path or str(Path(model_name_or_path).parent)
    )
    device = choose_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        cache_dir=str(args.cache_dir),
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=str(args.cache_dir))
    configure_mbart_language(model, tokenizer, args.src_lang, args.tgt_lang)
    model.to(device)
    model.eval()

    predictions = generate_corrections(
        model,
        tokenizer,
        device,
        inputs,
        args.batch_size,
        args.max_source_length,
        args.max_target_length,
        args.num_beams,
    )

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    else:
        for prediction in predictions:
            print(prediction)


if __name__ == "__main__":
    main()
