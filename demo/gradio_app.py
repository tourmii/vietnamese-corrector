#!/usr/bin/env python3
"""
app_compare.py — Vietnamese Text Correction: Parallel Model Comparison

All five models run in separate threads simultaneously.
Results stream into the UI as each model finishes, so you can compare
outputs, latency, and quality side-by-side.

Models
------
HuggingFace (auto-downloaded from Hub):
  • MinhDucNguyen9705/vietnamese-correction-2.0
  • tourmii/t5-vietnamese-corrector
  • khangdoan/mbart-vi-ocr-adaptation

Custom local checkpoints (optional — shown as "not configured" if omitted):
  • transformer_moe  →  --transformer-moe-ckpt
  • bilstm           →  --bilstm-model + --bilstm-src-vocab
                        + --bilstm-tgt-vocab + --bilstm-lm

Quick start
-----------
  # HuggingFace models only
  python app_compare.py

  # All five models
  python app_compare.py \\
      --transformer-moe-ckpt models/transformer_moe/checkpoints/epoch_04.pt \\
      --bilstm-dir            /path/to/bilstm_project \\
      --bilstm-module         predict \\
      --bilstm-model          /path/to/bilstm_project/checkpoint/bilstm_model \\
      --bilstm-src-vocab      /path/to/bilstm_project/checkpoint/vocab.src \\
      --bilstm-tgt-vocab      /path/to/bilstm_project/checkpoint/vocab.tgt \\
      --bilstm-lm             /path/to/bilstm_project/lm/corpus-wplm-4g-v2.binary \\
      --share
"""

from __future__ import annotations

import argparse
import importlib
import queue
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HF_MODELS = [
    "MinhDucNguyen9705/vietnamese-correction-2.0",
    "tourmii/t5-vietnamese-corrector",
    "khangdoan/mbart-vi-ocr-adaptation",
]
CUSTOM_MODELS = ["transformer_moe", "bilstm"]
ALL_MODELS = HF_MODELS + CUSTOM_MODELS

SHORT_NAMES: dict[str, str] = {
    "MinhDucNguyen9705/vietnamese-correction-2.0": "ViCorrect 2.0",
    "tourmii/t5-vietnamese-corrector":             "T5 Corrector",
    "khangdoan/mbart-vi-ocr-adaptation":           "mBART OCR",
    "transformer_moe":                             "Transformer MoE",
    "bilstm":                                      "BiLSTM",
}

T5_PROMPT_MODELS: frozenset[str] = frozenset({"tourmii/t5-vietnamese-corrector"})
T5_PREFIX = "Correct the grammatical errors in the following sentence.\n\n"
T5_SUFFIX = "\n\nCorrection: "

EXAMPLES = [
    "t đang xu ly 1 bai toán la sưa lỗi cho tieng viet",
    "hom nay troi dep wa nen minh muon di choi",
    "toi dang hoc xu ly ngon ngu tu nhien",
    "ban co the giup toi sua loi chinh ta khong",
    "nhung nguoi ban cua toi rat tot bung",
]


# ---------------------------------------------------------------------------
# Global lazy state
# ---------------------------------------------------------------------------

_args: Optional[argparse.Namespace] = None

# TransformerMoE: (model, tokenizer, model_cfg, correct_one_fn)
_moe_bundle: Optional[tuple] = None
_moe_lock = threading.Lock()

# BiLSTM: SamplePredictor instance
_bilstm: Optional[object] = None
_bilstm_lock = threading.Lock()


# ---------------------------------------------------------------------------
# HuggingFace models
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_hf(model_name: str, device: str):
    """Load and cache a HuggingFace seq2seq model."""
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()
    return tokenizer, model


def _hf_prompt(text: str, model_name: str) -> str:
    text = text.strip()
    return (T5_PREFIX + text + T5_SUFFIX) if model_name in T5_PROMPT_MODELS else text


def infer_hf(
    text: str,
    model_name: str,
    device: str,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    tokenizer, model = _load_hf(model_name, device)
    enc = tokenizer(
        _hf_prompt(text, model_name),
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    with torch.inference_mode():
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# TransformerMoE
# Delegates to load_model() / correct_one() from infer.py (document 1).
# ---------------------------------------------------------------------------

def _ensure_moe() -> tuple:
    """Lazy-load the TransformerMoE bundle (thread-safe)."""
    global _moe_bundle
    if _moe_bundle is not None:
        return _moe_bundle

    with _moe_lock:
        if _moe_bundle is not None:
            return _moe_bundle

        if not (_args and _args.transformer_moe_ckpt):
            raise RuntimeError(
                "TransformerMoE not configured — pass "
                "--transformer-moe-ckpt /path/to/checkpoint.pt"
            )

        ckpt = Path(_args.transformer_moe_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        # Directory layout assumed:
        #   <project_root>/
        #     infer.py, model.py, config.py   ← importable from here
        #     models/transformer_moe/checkpoints/epoch_XX.pt
        # If your layout differs, adjust parents[N] accordingly.
        if _args.transformer_moe_dir:
            project_root = _args.transformer_moe_dir
        else:
            # Walk up: checkpoints/ → transformer_moe/ → models/ → project root
            project_root = str(ckpt.parents[3])

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Import from infer.py as described in the inference script (document 1)
        from infer import correct_one, load_model  # type: ignore[import]

        device = torch.device(_args.device)
        model, tokenizer, model_cfg = load_model(str(ckpt), device)
        _moe_bundle = (model, tokenizer, model_cfg, correct_one)

    return _moe_bundle


def infer_transformer_moe(text: str) -> str:
    """Run TransformerMoE correction via correct_one() from infer.py."""
    model, tokenizer, model_cfg, correct_one = _ensure_moe()
    # correct_one is decorated @torch.no_grad() — safe to call from multiple threads
    return correct_one(text, model, tokenizer, model_cfg)


# ---------------------------------------------------------------------------
# BiLSTM
# Delegates to SamplePredictor from the bilstm inference module (document 3).
# ---------------------------------------------------------------------------

def _ensure_bilstm():
    """Lazy-load the BiLSTM SamplePredictor (thread-safe)."""
    global _bilstm
    if _bilstm is not None:
        return _bilstm

    with _bilstm_lock:
        if _bilstm is not None:
            return _bilstm

        missing = [
            f for f in ("bilstm_model", "bilstm_src_vocab", "bilstm_tgt_vocab", "bilstm_lm")
            if not getattr(_args, f, None)
        ]
        if not _args or missing:
            raise RuntimeError(
                "BiLSTM not configured — pass: "
                "--bilstm-model, --bilstm-src-vocab, --bilstm-tgt-vocab, --bilstm-lm"
            )

        # Add bilstm project directory to sys.path so its internal imports resolve
        bilstm_dir = (
            _args.bilstm_dir
            or str(Path(_args.bilstm_model).parents[1])
        )
        if bilstm_dir not in sys.path:
            sys.path.insert(0, bilstm_dir)

        # Import SamplePredictor from the configured module
        # Default module name is "predict" — override with --bilstm-module
        module_name = getattr(_args, "bilstm_module", "predict") or "predict"
        mod = importlib.import_module(module_name)
        SamplePredictor = mod.SamplePredictor  # noqa: N806

        _bilstm = SamplePredictor(
            src_vocab_path=_args.bilstm_src_vocab,
            tgt_vocab_path=_args.bilstm_tgt_vocab,
            model_path=_args.bilstm_model,
            wlm_path=_args.bilstm_lm,
        )

    return _bilstm


def infer_bilstm(text: str) -> str:
    """Run BiLSTM correction via SamplePredictor.predict_sample() (document 3)."""
    predictor = _ensure_bilstm()
    # predict_sample uses torch.no_grad() and only reads shared state → thread-safe
    return predictor.predict_sample(text)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def infer_model(
    text: str,
    model_name: str,
    device: str,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    if model_name in HF_MODELS:
        return infer_hf(text, model_name, device, max_input_length, max_new_tokens, num_beams)
    if model_name == "transformer_moe":
        return infer_transformer_moe(text)
    if model_name == "bilstm":
        return infer_bilstm(text)
    raise ValueError(f"Unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# Parallel streaming
# ---------------------------------------------------------------------------

_ST_RUNNING = "running"
_ST_DONE    = "done"
_ST_SKIP    = "skip"
_ST_ERROR   = "error"


def _fmt(status: str, output: str, ms: Optional[float]) -> str:
    """Format one model result for a Gradio Textbox."""
    if status == _ST_RUNNING:
        return "⏳  Running…"
    if status == _ST_SKIP:
        return f"⏭️  Not configured\n\n{output}"
    if status == _ST_ERROR:
        return f"❌  Error\n\n{output}"
    time_str = f"\n\n{'─' * 30}\n⏱  {ms:.0f} ms" if ms is not None else ""
    return f"{output}{time_str}"


def run_all_parallel(
    text: str,
    device: str,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> Generator[tuple[str, ...], None, None]:
    """
    Gradio generator that launches all five models in threads and
    yields an updated 5-tuple of formatted strings every time one finishes.
    """
    if not (text and text.strip()):
        yield tuple("" for _ in ALL_MODELS)
        return

    # State map: model → (status, output, elapsed_ms)
    states: dict[str, tuple[str, str, Optional[float]]] = {
        m: (_ST_RUNNING, "", None) for m in ALL_MODELS
    }
    result_q: queue.Queue[tuple[str, str, str, Optional[float]]] = queue.Queue()

    def _worker(model_name: str) -> None:
        t0 = time.perf_counter()
        try:
            out = infer_model(
                text, model_name, device,
                int(max_input_length), int(max_new_tokens), int(num_beams),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result_q.put((model_name, _ST_DONE, out, elapsed_ms))
        except RuntimeError as exc:
            tag = _ST_SKIP if "not configured" in str(exc).lower() else _ST_ERROR
            result_q.put((model_name, tag, str(exc), None))
        except Exception as exc:
            result_q.put((model_name, _ST_ERROR, str(exc), None))

    threads = [
        threading.Thread(target=_worker, args=(m,), daemon=True)
        for m in ALL_MODELS
    ]
    for t in threads:
        t.start()

    # Immediately yield the "all running" state
    yield tuple(_fmt(*states[m]) for m in ALL_MODELS)

    # Yield an updated tuple each time a model completes
    for _ in ALL_MODELS:
        name, status, out, ms = result_q.get()
        states[name] = (status, out, ms)
        yield tuple(_fmt(*states[m]) for m in ALL_MODELS)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
/* Card borders around each result column */
.model-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 14px 16px 10px;
    background: var(--background-fill-primary);
}
/* Monospace output for easy diffing */
.model-card textarea {
    font-family: 'Menlo', 'Consolas', 'Courier New', monospace !important;
    font-size: 0.875em !important;
    line-height: 1.6 !important;
}
/* Model name header inside each card */
.model-name {
    font-size: 0.82em;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.7;
}
footer { display: none !important; }
"""


def _card(model_name: str) -> tuple[gr.Markdown, gr.Textbox]:
    """Build one model result card, return (label, textbox)."""
    with gr.Column(elem_classes="model-card"):
        label = gr.Markdown(f"**{SHORT_NAMES[model_name]}**", elem_classes="model-name")
        box = gr.Textbox(
            show_label=False,
            lines=7,
            interactive=False,
            placeholder="Waiting…",
        )
    return label, box


def build_app(args: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(
        title="🇻🇳 Vietnamese Correction — Parallel Comparison",
        css=CSS,
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "## 🇻🇳 Vietnamese Text Correction — Parallel Model Comparison\n"
            "All five models run simultaneously in separate threads. "
            "Results stream in as each model finishes."
        )

        # ── Input & controls ─────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=5):
                input_box = gr.Textbox(
                    label="Input sentence",
                    placeholder="Nhập câu tiếng Việt cần sửa…",
                    lines=3,
                )
            with gr.Column(scale=1, min_width=170):
                correct_btn = gr.Button("▶  Correct All", variant="primary", size="lg")
                with gr.Accordion("⚙  Generation settings", open=False):
                    max_input_length = gr.Slider(
                        32, 512, step=16, value=args.max_input_length, label="Max input len"
                    )
                    max_new_tokens = gr.Slider(
                        16, 256, step=8, value=args.max_new_tokens, label="Max new tokens"
                    )
                    num_beams = gr.Slider(
                        1, 8, step=1, value=args.num_beams, label="Beam size"
                    )

        # ── Result cards: row of 3, then row of 2 ───────────────────────────
        outputs: list[gr.Textbox] = []

        with gr.Row():
            for m in ALL_MODELS[:3]:
                _, box = _card(m)
                outputs.append(box)

        with gr.Row():
            gr.Column(scale=1)          # left spacer for visual centering
            for m in ALL_MODELS[3:]:
                with gr.Column(scale=2, elem_classes="model-card"):
                    gr.Markdown(f"**{SHORT_NAMES[m]}**", elem_classes="model-name")
                    box = gr.Textbox(
                        show_label=False,
                        lines=7,
                        interactive=False,
                        placeholder="Waiting…",
                    )
                    outputs.append(box)
            gr.Column(scale=1)          # right spacer

        # ── Wiring ───────────────────────────────────────────────────────────
        device_state = gr.State(args.device)
        fn_inputs = [input_box, device_state, max_input_length, max_new_tokens, num_beams]

        # Button click → run all models in parallel, stream results
        correct_btn.click(
            fn=run_all_parallel,
            inputs=fn_inputs,
            outputs=outputs,
        )
        # Pressing Enter in the input box also triggers inference
        input_box.submit(
            fn=run_all_parallel,
            inputs=fn_inputs,
            outputs=outputs,
        )

        # ── Examples ─────────────────────────────────────────────────────────
        gr.Examples(
            examples=[[e] for e in EXAMPLES],
            inputs=[input_box],
            label="📝 Example sentences (click to load, then press Enter or the button)",
        )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel Vietnamese text correction: compare all 5 models side-by-side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7860)
    p.add_argument("--share",       action="store_true", help="Create a public Gradio link")

    # Generation defaults (for HuggingFace models)
    p.add_argument("--max-input-length", type=int, default=128)
    p.add_argument("--max-new-tokens",   type=int, default=128)
    p.add_argument("--num-beams",        type=int, default=4)

    # ── TransformerMoE ───────────────────────────────────────────────────────
    moe = p.add_argument_group("TransformerMoE (infer.py)")
    moe.add_argument(
        "--transformer-moe-ckpt",
        default="/home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/transformer_moe/checkpoints/epoch_04.pt",
        metavar="PATH",
        help="Path to TransformerMoE .pt checkpoint (e.g. models/transformer_moe/checkpoints/epoch_04.pt)",
    )
    moe.add_argument(
        "--transformer-moe-dir",
        default="/home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/transformer_moe",
        metavar="DIR",
        help="Project root containing infer.py/model.py/config.py "
             "(auto-inferred from ckpt path if omitted)",
    )

    # ── BiLSTM ───────────────────────────────────────────────────────────────
    bilstm = p.add_argument_group("BiLSTM (SamplePredictor)")
    bilstm.add_argument(
        "--bilstm-dir",
        default="/home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/bilstm",
        metavar="DIR",
        help="BiLSTM project root added to sys.path for internal imports",
    )
    bilstm.add_argument(
        "--bilstm-module",
        default="predict",
        metavar="MODULE",
        help="Python module name that contains SamplePredictor (e.g. 'predict')",
    )
    bilstm.add_argument("--bilstm-model",     default="/home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/bilstm/checkpoint/bilstm_model.ep25", metavar="PATH",
                        help="Path to BiLSTM model checkpoint")
    bilstm.add_argument("--bilstm-src-vocab", default="/home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/bilstm/checkpoint/vocab.src", metavar="PATH",
                        help="Path to source vocab (.src)")
    bilstm.add_argument("--bilstm-tgt-vocab", default="/home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/bilstm/checkpoint/vocab.tgt", metavar="PATH",
                        help="Path to target vocab (.tgt)")
    bilstm.add_argument("--bilstm-lm",        default="/home/s/sangdv_student/ethnic_s2t/text2text_data/vietnamese-corrector/models/bilstm/checkpoint/lm.binary", metavar="PATH",
                        help="Path to KenLM word language model (.binary)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _args
    _args = parse_args()

    print(f"Device : {_args.device}")
    print(f"MoE ckpt: {_args.transformer_moe_ckpt or '(not configured)'}")
    print(f"BiLSTM  : {_args.bilstm_model or '(not configured)'}\n")

    app = build_app(_args)

    # Queue with enough concurrency for 5 parallel model threads per request
    try:
        app = app.queue(default_concurrency_limit=4)
    except TypeError:
        app = app.queue(concurrency_count=4)

    app.launch(
        server_name=_args.server_name,
        server_port=_args.server_port,
        share=True
    )


if __name__ == "__main__":
    main()
