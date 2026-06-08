import argparse
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS_DIR))

from gen_error import ALL_ERROR_TYPES, generate_error  # noqa: E402


DEFAULT_MODELS = [
    "MinhDucNguyen9705/vietnamese-correction-2.0",
    "tourmii/t5-vietnamese-corrector",
    "khangdoan/mbart-vi-ocr-adaptation",
    "transformer_moe"
]

T5_PROMPT_MODELS = {"tourmii/t5-vietnamese-corrector"}
PROMPT_PREFIX = "Correct the grammatical errors in the following sentence.\n\n"
PROMPT_SUFFIX = "\n\nCorrection: "


def build_model_input(text: str, model_name: str) -> str:
    text = text.strip()
    if model_name in T5_PROMPT_MODELS:
        return PROMPT_PREFIX + text + PROMPT_SUFFIX
    return text


def generate_noisy_text(text: str, error_type: str) -> tuple[str, str]:
    if not text or not text.strip():
        return "", ""

    selected_error_type = None if error_type == "random" else error_type
    noisy, used_error_type = generate_error(text.strip(), error_type=selected_error_type)
    return noisy, used_error_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Gradio demo for Vietnamese text correction.")
    parser.add_argument("--model", default=DEFAULT_MODELS[0], help="Default Hugging Face model ID/path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    return parser.parse_args()


def torch_dtype_for(device: str):
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


@lru_cache(maxsize=4)
def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype_for(device),
    ).to(device)
    model.eval()
    return tokenizer, model


def correct_text(
    text: str,
    model_name: str,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    device: str,
) -> str:
    if not text or not text.strip():
        return ""

    tokenizer, model = load_model(model_name, device)
    encoded = tokenizer(
        build_model_input(text, model_name),
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def correct_stream(
    text: str,
    model_name: str,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    device: str,
):
    if not text or not text.strip():
        yield ""
        return

    yield "Correcting..."
    try:
        yield correct_text(
            text=text,
            model_name=model_name,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            device=device,
        )
    except Exception as exc:
        yield f"Error: {exc}"


def add_live_event(textbox: gr.Textbox, fn, inputs: list, outputs: gr.Textbox) -> None:
    try:
        textbox.input(
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            show_progress="hidden",
            trigger_mode="always_last",
        )
    except TypeError:
        textbox.input(
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            show_progress="hidden",
        )


def build_app(args: argparse.Namespace) -> gr.Blocks:
    default_model = args.model if args.model in DEFAULT_MODELS else args.model
    model_choices = DEFAULT_MODELS if args.model in DEFAULT_MODELS else [args.model, *DEFAULT_MODELS]

    with gr.Blocks(title="Vietnamese Correction") as demo:
        gr.Markdown("# Vietnamese Correction")

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Model",
                )
                error_type_dropdown = gr.Dropdown(
                    choices=["random", *ALL_ERROR_TYPES],
                    value="random",
                    label="Error type",
                )
                generate_error_button = gr.Button("Generate error", variant="secondary")
                generated_error_type = gr.Textbox(
                    label="Generated error type",
                    interactive=False,
                )
                max_input_length = gr.Slider(
                    minimum=32,
                    maximum=512,
                    step=16,
                    value=args.max_input_length,
                    label="Max input length",
                )
                max_new_tokens = gr.Slider(
                    minimum=16,
                    maximum=256,
                    step=8,
                    value=args.max_new_tokens,
                    label="Max new tokens",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=args.num_beams,
                    label="Beam size",
                )

            with gr.Column(scale=2):
                source = gr.Textbox(
                    label="Input",
                    placeholder="Nhap cau tieng Viet can sua...",
                    lines=8,
                )
                correction = gr.Textbox(
                    label="Correction",
                    lines=8,
                    interactive=False,
                )

        device_state = gr.State(args.device)
        inputs = [
            source,
            model_dropdown,
            max_input_length,
            max_new_tokens,
            num_beams,
            device_state,
        ]

        add_live_event(source, correct_stream, inputs, correction)
        generate_error_button.click(
            fn=generate_noisy_text,
            inputs=[source, error_type_dropdown],
            outputs=[source, generated_error_type],
            show_progress="hidden",
        ).then(
            fn=correct_stream,
            inputs=inputs,
            outputs=correction,
            show_progress="hidden",
        )
        for control in (model_dropdown, max_input_length, max_new_tokens, num_beams):
            control.change(
                fn=correct_stream,
                inputs=inputs,
                outputs=correction,
                show_progress="hidden",
            )

        gr.Examples(
            examples=[
                ["t đang xu ly 1 bai toán la sưa lỗi cho tieng viet"],
                ["hom nay troi dep wa nen minh muon di choi"],
                ["toi dang hoc xu ly ngon ngu tu nhien"],
            ],
            inputs=source,
        )

    return demo


def queue_app(app: gr.Blocks) -> gr.Blocks:
    try:
        return app.queue(default_concurrency_limit=1)
    except TypeError:
        return app.queue(concurrency_count=1)


def main() -> None:
    args = parse_args()
    app = build_app(args)
    queue_app(app).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
