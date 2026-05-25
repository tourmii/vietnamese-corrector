import random
import re
from typing import Callable


from abbreviation import abbreviate
from teencode import teencode
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
    "no_diacritics_abbreviation",
    "no_diacritics_region",
]


def no_diacritics_abbreviation(text: str) -> str:
    return abbreviate(no_diacritic(text))


def no_diacritics_region(text: str) -> str:
    return region_error(no_diacritic(text), dialect="random")


def _apply_tokenwise(
    text: str, fn: Callable[[str], str], token_rate: float, rng: random.Random
) -> str:
    parts = re.split(r"(\s+)", text)
    for idx, token in enumerate(parts):
        if token.isspace() or not token:
            continue
        if rng.random() < token_rate:
            parts[idx] = fn(token)
    return "".join(parts)


def _random_kwargs(error_type: ErrorType, rng: random.Random, **kwargs) -> dict:
    params: dict = {}
    if error_type == "teencode":
        params["intensity"] = kwargs.get("intensity", rng.uniform(0.35, 0.75))
    elif error_type == "fat_finger":
        params["error_rate"] = kwargs.get("error_rate", rng.uniform(0.06, 0.2))
    elif error_type == "telex":
        params["mode"] = kwargs.get("mode", "random")
        params["token_rate"] = kwargs.get("token_rate", rng.uniform(0.2, 0.5))
    elif error_type == "region":
        params["dialect"] = kwargs.get("dialect", "random")
    elif error_type == "edit_distance":
        params["num_edits"] = kwargs.get("num_edits", rng.choice([1, 1, 2]))
    return params


def generate_error(
    text: str,
    error_type: ErrorType | None = None,
    **kwargs,
) -> tuple[str, ErrorType]:
    rng = random.Random(kwargs.pop("seed", None))
    if error_type is None:
        error_type = random.choice(ALL_ERROR_TYPES)

    dispatch: dict[ErrorType, Callable[..., str]] = {
        "abbreviation": abbreviate,
        "teencode": teencode,
        "fat_finger": fat_finger,
        "telex": telex_error,
        "region": region_error,
        "edit_distance": edit_distance_error,
        "no_diacritic": no_diacritic,
        "no_diacritics_abbreviation": no_diacritics_abbreviation,
        "no_diacritics_region": no_diacritics_region,
    }

    if error_type not in dispatch:
        raise ValueError(f"Unknown error_type '{error_type}'. Choose from {ALL_ERROR_TYPES}")

    fn = dispatch[error_type]
    params = _random_kwargs(error_type, rng, **kwargs)

    if error_type == "telex":
        token_rate = params.pop("token_rate")
        noisy = _apply_tokenwise(text, lambda t: fn(t, **params), token_rate, rng)
    else:
        noisy = fn(text, **params)
    return noisy, error_type


def generate_all_errors(text: str):
    return {et: generate_error(text, et)[0] for et in ALL_ERROR_TYPES}


if __name__ == "__main__":
    samples = [
        "Không biết làm thế nào",
        "Bình thường thôi",
        "Xin chào mọi người",
        "Điện thoại của tôi bị hỏng",
        "Trường học rất đẹp",
        "ừ nó cười kiểu mỉa mai",
    ]

    for text in samples:
        print(f"\nOriginal : {text}")
        for et, noisy in generate_all_errors(text).items():
            print(f"  {et:<16}: {noisy}")

    for text in samples:
        print(generate_error(text, error_type='teencode', intensity=1))