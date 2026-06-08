"""
infer.py — Vietnamese Text Correction Inference

Modes:
  --text   "input string"          correct a single sentence
  --file   input.txt               correct every line, write to output.txt (--output)
  --interactive                    REPL loop

Usage:
  python infer.py --ckpt /home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/transformer_moe/checkpoints/epoch_04.pt --interactive
"""

import argparse
import sys
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from config import ModelConfig
from model import TransformerMoE


# ── loader ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    tokenizer_name = ckpt.get("tokenizer_name", "vinai/bartpho-syllable-base")

    if "model_cfg" in ckpt:
        model_cfg = ckpt["model_cfg"]
    else:
        # infer critical dims from saved weight shapes to avoid size mismatch
        sd = ckpt["model"]
        vocab_size, d_model = sd["src_embed.weight"].shape
        model_cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model)
        print(f"[warn] model_cfg not in checkpoint — inferred vocab_size={vocab_size}, d_model={d_model}")

    model = TransformerMoE(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    step = ckpt.get("step", "?")
    bleu = ckpt.get("best_bleu", None)
    bleu_str = f"  best_bleu={bleu:.2f}" if bleu is not None else ""
    print(f"Loaded  : {ckpt_path}  (step={step}{bleu_str})")
    print(f"Tokenizer: {tokenizer_name}  |  vocab={model_cfg.vocab_size}")
    print(f"Device  : {device}\n")

    return model, tokenizer, model_cfg


# ── core inference ─────────────────────────────────────────────────────────────

def _tokenize_batch(texts: list[str], tokenizer, max_len: int, device: torch.device):
    ids = [
        torch.tensor(
            tokenizer(t, max_length=max_len, truncation=True, padding=False)["input_ids"]
        )
        for t in texts
    ]
    src = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    src_mask = (src == tokenizer.pad_token_id).to(device)
    return src, src_mask


@torch.no_grad()
def correct_batch(
    texts: list[str],
    model: TransformerMoE,
    tokenizer,
    model_cfg: ModelConfig,
    batch_size: int = 32,
) -> list[str]:
    device = next(model.parameters()).device
    results = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        src, src_mask = _tokenize_batch(chunk, tokenizer, model_cfg.max_seq_len, device)
        preds = model.generate(src, src_mask, max_len=model_cfg.max_seq_len)
        for pred in preds:
            results.append(tokenizer.decode(pred.tolist(), skip_special_tokens=True))

    return results


def correct_one(
    text: str,
    model: TransformerMoE,
    tokenizer,
    model_cfg: ModelConfig,
) -> str:
    return correct_batch([text], model, tokenizer, model_cfg)[0]


# ── modes ─────────────────────────────────────────────────────────────────────

def run_single(args, model, tokenizer, model_cfg):
    t0 = time.perf_counter()
    out = correct_one(args.text, model, tokenizer, model_cfg)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Input  : {args.text}")
    print(f"Output : {out}")
    print(f"({ms:.1f} ms)")


def run_file(args, model, tokenizer, model_cfg):
    with open(args.file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    print(f"Correcting {len(lines):,} lines  (batch={args.batch_size}) ...")
    t0 = time.perf_counter()
    outputs = correct_batch(lines, model, tokenizer, model_cfg, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0

    out_path = args.output or (args.file.rsplit(".", 1)[0] + "_corrected.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(outputs) + "\n")

    print(f"Saved  : {out_path}")
    print(f"Speed  : {len(lines) / elapsed:.0f} lines/s  ({elapsed:.1f}s total)")


def run_interactive(args, model, tokenizer, model_cfg):
    print("Interactive mode — type a sentence to correct, 'q' to quit.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            break
        t0 = time.perf_counter()
        out = correct_one(text, model, tokenizer, model_cfg)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    {out}  ({ms:.0f}ms)\n")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vietnamese text correction inference")
    parser.add_argument("--ckpt",        required=True,       help="Path to checkpoint (.pt)")
    parser.add_argument("--tokenizer",   default=None,        help="Override tokenizer name/path")
    parser.add_argument("--device",      default=None,        help="cpu | cuda | cuda:0 (auto-detect if omitted)")
    parser.add_argument("--batch-size",  type=int, default=32, help="Batch size for --file mode")
    parser.add_argument("--max-len",     type=int, default=None, help="Override max generation length")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text",        type=str,  help="Single input sentence")
    mode.add_argument("--file",        type=str,  help="Input .txt file (one sentence per line)")
    mode.add_argument("--interactive", action="store_true", help="Interactive REPL")

    parser.add_argument("--output", type=str, default=None,
                        help="Output path for --file mode (default: <input>_corrected.txt)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, model_cfg = load_model(args.ckpt, device)

    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.max_len:
        model_cfg.max_seq_len = args.max_len

    if args.text:
        run_single(args, model, tokenizer, model_cfg)
    elif args.file:
        run_file(args, model, tokenizer, model_cfg)
    else:
        run_interactive(args, model, tokenizer, model_cfg)


if __name__ == "__main__":
    main()