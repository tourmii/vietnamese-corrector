import random
import re

REGION_SUBS_SOUTH_START: list[tuple[str, str]] = [
    ("ch", "c"), ("tr", "ch"), ("gi", "d"),
    ("x", "s"), ("s", "x"), ("kh", "h")
]

REGION_SUBS_SOUTH_END: list[tuple[str, str]] = [
    ("n", "ng"),
    ("t", "c"), ("c", "t"),
]

REGION_SUBS_SOUTH_ANY: list[tuple[str, str]] = [
    ("v", "d"),
    ("ươ", "ưa"), ("iê", "ia"), ("uô", "ua"),
    ("ê", "e"), ("â", "ơ"),
    ("ã", "ạ"), ("ẫ", "ậ"), ("ễ", "ệ"), ("ĩ", "ị"), ("ỗ", "ộ"), ("ữ", "ự"), ("ỹ", "ỵ"),
]

REGION_SUBS_NORTH_START: list[tuple[str, str]] = [
    ("l", "n"), ("n", "l"),
    ("r", "d"),
    ("tr", "ch"), ("ch", "tr"),
]

REGION_SUBS_NORTH_END: list[tuple[str, str]] = []

REGION_SUBS_NORTH_ANY: list[tuple[str, str]] = []


def region_error(text: str, dialect: str = "south") -> str:
    if dialect == "random":
        dialect = random.choice(["south", "north"])

    subs_any = REGION_SUBS_SOUTH_ANY if dialect == "south" else REGION_SUBS_NORTH_ANY
    result = text
    for src, dst in subs_any:
        if random.random() < 0.5:
            result = result.replace(src, dst)
            result = result.replace(src.upper(), dst.upper())
            result = result.replace(src.capitalize(), dst.capitalize())

    def boundary_replace(result: str, src: str, dst: str, prefix: str, suffix: str) -> str:
        variants = [
            (src, dst),
            (src.upper(), dst.upper()),
            (src.capitalize(), dst.capitalize()),
        ]
        for src_variant, dst_variant in variants:
            pattern = f"{prefix}{re.escape(src_variant)}{suffix}"
            result = re.sub(pattern, dst_variant, result)
        return result

    if dialect == "south":
        for src, dst in REGION_SUBS_SOUTH_START:
            if random.random() < 0.5:
                result = boundary_replace(result, src, dst, r"\b", "")

        for src, dst in REGION_SUBS_SOUTH_END:
            if random.random() < 0.5:
                result = boundary_replace(result, src, dst, "", r"\b")

    if dialect == "north":
        for src, dst in REGION_SUBS_NORTH_START:
            if random.random() < 0.5:
                result = boundary_replace(result, src, dst, r"\b", "")

        for src, dst in REGION_SUBS_NORTH_END:
            if random.random() < 0.5:
                result = boundary_replace(result, src, dst, "", r"\b")
    return result


__all__ = [
    "REGION_SUBS_SOUTH_ANY",
    "REGION_SUBS_SOUTH_START",
    "REGION_SUBS_SOUTH_END",
    "REGION_SUBS_NORTH_ANY",
    "REGION_SUBS_NORTH_START",
    "REGION_SUBS_NORTH_END",
    "region_error",
]
