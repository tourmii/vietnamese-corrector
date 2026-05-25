import random
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
    "no_diacritic",
]


def generate_error(
    text: str,
    error_type: ErrorType | None = None,
    **kwargs,
) -> tuple[str, ErrorType]:
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
    }

    if error_type not in dispatch:
        raise ValueError(f"Unknown error_type '{error_type}'. Choose from {ALL_ERROR_TYPES}")

    fn = dispatch[error_type]
    noisy = fn(text, **kwargs)
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