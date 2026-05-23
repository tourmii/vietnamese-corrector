import random

TEEN_SUBS: list[tuple[str, str]] = [
    ("ươ", "uo"),
    ("iê", "ie"),
    ("uô", "uo"),
    ("ph", "f"),
    ("đ", "d"),
    ("ă", "a"),
    ("â", "a"),
    ("ê", "e"),
    ("ô", "o"),
    ("ơ", "o"),
    ("ư", "u"),
    ("à", "a`"), ("ả", "a?"), ("ã", "a~"), ("á", "a'"), ("ạ", "a."),
    ("ề", "e`"), ("ể", "e?"), ("ễ", "e~"), ("ế", "e'"), ("ệ", "e."),
    ("ì", "i`"), ("ỉ", "i?"), ("ĩ", "i~"), ("í", "i'"), ("ị", "i."),
    ("ò", "o`"), ("ỏ", "o?"), ("õ", "o~"), ("ó", "o'"), ("ọ", "o."),
    ("ù", "u`"), ("ủ", "u?"), ("ũ", "u~"), ("ú", "u'"), ("ụ", "u."),
    ("ỳ", "y`"), ("ỷ", "y?"), ("ỹ", "y~"), ("ý", "y'"), ("ỵ", "y."),
    ("o", "0"), ("i", "1"), ("e", "3"), ("a", "4"),
]

TEEN_INSERTIONS: list[str] = ["j", "z", "w", "x", "q"]


def teencode(text: str, intensity: float = 0.6) -> str:
    result = text
    for src, dst in TEEN_SUBS:
        if random.random() < intensity:
            result = result.replace(src, dst)
            result = result.replace(src.upper(), dst.upper())
    if random.random() < intensity * 0.3:
        pos = random.randint(0, len(result))
        result = result[:pos] + random.choice(TEEN_INSERTIONS) + result[pos:]
    return result


__all__ = ["TEEN_SUBS", "TEEN_INSERTIONS", "teencode"]
