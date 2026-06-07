"""Shared mBART tokenizer/model setup."""

from __future__ import annotations

from typing import Any


def configure_mbart_language(model: Any, tokenizer: Any, src_lang: str, tgt_lang: str) -> None:
    if not hasattr(tokenizer, "lang_code_to_id"):
        raise ValueError("The selected tokenizer does not expose mBART language codes.")
    if src_lang not in tokenizer.lang_code_to_id:
        raise ValueError(f"Unknown source language {src_lang!r}. Available examples: {list(tokenizer.lang_code_to_id)[:10]}")
    if tgt_lang not in tokenizer.lang_code_to_id:
        raise ValueError(f"Unknown target language {tgt_lang!r}. Available examples: {list(tokenizer.lang_code_to_id)[:10]}")

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    target_lang_id = tokenizer.lang_code_to_id[tgt_lang]

    tokenizer_class_name = tokenizer.__class__.__name__.lower()
    model_name = getattr(model.config, "_name_or_path", "").lower()
    is_mbart50 = "mbart50" in tokenizer_class_name or "mbart-large-50" in model_name

    if is_mbart50:
        model.config.decoder_start_token_id = tokenizer.eos_token_id
        model.config.forced_bos_token_id = target_lang_id
        model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
        model.generation_config.forced_bos_token_id = target_lang_id
    else:
        model.config.decoder_start_token_id = target_lang_id
        model.generation_config.decoder_start_token_id = target_lang_id
        model.config.forced_bos_token_id = None
        model.generation_config.forced_bos_token_id = None
