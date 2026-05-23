import random

REGION_SUBS_SOUTH: list[tuple[str, str]] = [
    ("ch", "c"), ("tr", "ch"), ("gi", "d"), ("v", "d"),
    ("x", "s"), ("s", "x"),
    ("n", "ng"), ("t", "c"), ("c", "t"),
    ("ươ", "ưa"), ("iê", "ia"), ("uô", "ua"),
    ("ê", "e"), ("â", "ơ"),
    ("ã", "ạ"), ("ẫ", "ậ"), ("ễ", "ệ"), ("ĩ", "ị"), ("ỗ", "ộ"), ("ữ", "ự"), ("ỹ", "ỵ"),
]

REGION_SUBS_NORTH: list[tuple[str, str]] = [
    ("l", "n"), ("n", "l"),
    ("r", "d"),
    ("tr", "ch"), ("ch", "tr"),
]


def region_error(text: str, dialect: str = "south") -> str:
    if dialect == "random":
        dialect = random.choice(["south", "north"])

    subs = REGION_SUBS_SOUTH if dialect == "south" else REGION_SUBS_NORTH
    result = text
    for src, dst in subs:
        if random.random() < 0.5:
            result = result.replace(src, dst)
            result = result.replace(src.upper(), dst.upper())
            result = result.replace(src.capitalize(), dst.capitalize())
    return result


__all__ = ["REGION_SUBS_SOUTH", "REGION_SUBS_NORTH", "region_error"]
