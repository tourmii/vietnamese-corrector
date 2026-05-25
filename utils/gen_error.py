from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Callable, Iterable, Optional


try:
    from .abbreviation import abbreviate
    from .teencode import TEEN_INSERTIONS, teencode
    from .fat_finger import fat_finger
    from .telex import telex_error
    from .region import region_error
    from .edit_distance import edit_distance_error
    from .no_diacritics import no_diacritic
except ImportError:
    from abbreviation import abbreviate
    from teencode import TEEN_INSERTIONS, teencode
    from fat_finger import fat_finger
    from telex import telex_error
    from region import region_error
    from edit_distance import edit_distance_error
    from no_diacritics import no_diacritic

ErrorType = str  

ALL_ERROR_TYPES: list[ErrorType] = [
    "abbreviation",
    "teencode",
    "fat_finger",
    "telex",
    "region",
    "edit_distance",
    "no_diacritic",
]

DISPATCH: dict[ErrorType, Callable[..., str]] = {
    "abbreviation": abbreviate,
    "teencode": teencode,
    "fat_finger": fat_finger,
    "telex": telex_error,
    "region": region_error,
    "edit_distance": edit_distance_error,
    "no_diacritic": no_diacritic,
}

GENERATION_KWARGS: dict[ErrorType, dict[str, object]] = {
    "teencode": {"intensity": 0.85},
    "fat_finger": {"error_rate": 0.25},
    "telex": {"mode": "random"},
    "region": {"dialect": "random"},
    "edit_distance": {"num_edits": 2},
}


def generate_error(
    text: str,
    error_type: Optional[ErrorType] = None,
    **kwargs,
) -> tuple[str, ErrorType]:
    if error_type is None:
        error_type = random.choice(ALL_ERROR_TYPES)

    if error_type not in DISPATCH:
        raise ValueError(f"Unknown error_type '{error_type}'. Choose from {ALL_ERROR_TYPES}")

    fn = DISPATCH[error_type]
    noisy = fn(text, **kwargs)
    return noisy, error_type


def generate_all_errors(text: str) -> dict[ErrorType, str]:
    return {et: generate_error(text, et)[0] for et in ALL_ERROR_TYPES}


def abbreviation_no_diacritic_error(text: str) -> str:
    abbreviated = abbreviate(text)
    noisy = no_diacritic(abbreviated)
    if abbreviated != text:
        return noisy

    return _fallback_abbreviation(no_diacritic(text))


def _fallback_abbreviation(text: str) -> str:
    match = re.search(r"\w{4,}", text)
    if match is None:
        return text

    word = match.group(0)
    vowels = set("aeiouAEIOU")
    shortened = word[0] + "".join(char for char in word[1:] if char not in vowels)
    if shortened == word:
        shortened = word[:-1]

    return text[: match.start()] + shortened + text[match.end() :]


def read_texts_from_csv(input_path: str | Path, text_column: Optional[str] = None) -> list[str]:
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.reader(f) if row and any(cell.strip() for cell in row)]

    if not rows:
        return []

    if text_column is not None:
        header = [cell.strip() for cell in rows[0]]
        if text_column not in header:
            raise ValueError(f"Column '{text_column}' not found. Available columns: {header}")
        col_idx = header.index(text_column)
        data_rows = rows[1:]
    elif len(rows[0]) == 1:
        col_idx = 0
        first_value = rows[0][0].strip().lower()
        data_rows = rows[1:] if first_value in {"0", "text", "sentence", "content"} else rows
    else:
        header = [cell.strip().lower() for cell in rows[0]]
        known_names = ["text", "sentence", "content", "clean_text"]
        col_idx = next((header.index(name) for name in known_names if name in header), 0)
        data_rows = rows[1:] if header[col_idx] in known_names else rows

    texts: list[str] = []
    for row in data_rows:
        if col_idx < len(row):
            text = row[col_idx].strip()
            if text:
                texts.append(text)
    return texts


def _generate_changed_error(
    text: str,
    error_type: ErrorType,
    max_attempts: int = 8,
) -> Optional[str]:
    kwargs = GENERATION_KWARGS.get(error_type, {})
    for _ in range(max_attempts):
        noisy, _ = generate_error(text, error_type=error_type, **kwargs)
        if noisy != text:
            return noisy

    if error_type == "teencode" and text:
        pos = random.randint(0, len(text))
        return text[:pos] + random.choice(TEEN_INSERTIONS) + text[pos:]

    return None


def _next_error_type(pool: list[ErrorType]) -> ErrorType:
    if not pool:
        pool.extend(ALL_ERROR_TYPES)
        random.shuffle(pool)
    return pool.pop()


def generate_error_rows(
    texts: Iterable[str],
    min_errors_per_sample: int = 2,
    max_errors_per_sample: int = 3,
    include_combo: bool = True,
    seed: Optional[int] = None,
) -> list[dict[str, object]]:
    if seed is not None:
        random.seed(seed)

    if min_errors_per_sample < 1:
        raise ValueError("min_errors_per_sample must be >= 1")
    if max_errors_per_sample < min_errors_per_sample:
        raise ValueError("max_errors_per_sample must be >= min_errors_per_sample")

    rows: list[dict[str, object]] = []
    error_pool: list[ErrorType] = []

    for sample_id, clean_text in enumerate(texts):
        target_count = random.randint(min_errors_per_sample, max_errors_per_sample)
        used_types: set[ErrorType] = set()
        variants: list[tuple[str, str]] = []
        attempts = 0

        while len(variants) < target_count and attempts < len(ALL_ERROR_TYPES) * 3:
            attempts += 1
            error_type = _next_error_type(error_pool)
            if error_type in used_types:
                continue

            noisy = _generate_changed_error(clean_text, error_type)
            if noisy is None:
                continue

            used_types.add(error_type)
            variants.append((noisy, error_type))

        while len(variants) < target_count and clean_text:
            noisy, error_type = generate_error(
                clean_text,
                error_type="edit_distance",
                num_edits=len(variants) + 1,
            )
            variants.append((noisy, error_type))

        if include_combo:
            variants.append((abbreviation_no_diacritic_error(clean_text), "abbreviation+no_diacritic"))

        for variant_id, (noisy_text, error_types) in enumerate(variants):
            rows.append(
                {
                    "sample_id": sample_id,
                    "variant_id": variant_id,
                    "clean_text": clean_text,
                    "noisy_text": noisy_text,
                    "error_types": error_types,
                }
            )

    return rows


def write_error_dataset(
    input_path: str | Path,
    output_path: str | Path,
    text_column: Optional[str] = None,
    min_errors_per_sample: int = 2,
    max_errors_per_sample: int = 3,
    include_combo: bool = True,
    seed: Optional[int] = None,
) -> tuple[int, int]:
    texts = read_texts_from_csv(input_path, text_column=text_column)
    rows = generate_error_rows(
        texts,
        min_errors_per_sample=min_errors_per_sample,
        max_errors_per_sample=max_errors_per_sample,
        include_combo=include_combo,
        seed=seed,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "variant_id", "clean_text", "noisy_text", "error_types"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(texts), len(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese spelling-error variants from a CSV file.",
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="dataset/1/test_dataset.csv",
        help="Input CSV. Defaults to dataset/1/test_dataset.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dataset/1/generated_errors.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Column name containing clean text. Omit for a one-column CSV.",
    )
    parser.add_argument("--min-errors", type=int, default=2)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-combo",
        action="store_true",
        help="Do not add the extra abbreviation+no_diacritic variant.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    text_count, row_count = write_error_dataset(
        input_path=args.input_csv,
        output_path=args.output,
        text_column=args.text_column,
        min_errors_per_sample=args.min_errors,
        max_errors_per_sample=args.max_errors,
        include_combo=not args.no_combo,
        seed=args.seed,
    )
    print(f"Wrote {row_count} noisy rows from {text_count} clean sentences to {args.output}")


if __name__ == "__main__":
    main()
