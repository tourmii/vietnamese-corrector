import random

QWERTY_ADJACENCY: dict[str, list[str]] = {
    "q": ["w", "a", "s"], "w": ["q", "e", "a", "s", "d"], "e": ["w", "r", "s", "d", "f"],
    "r": ["e", "t", "d", "f", "g"], "t": ["r", "y", "f", "g", "h"],
    "y": ["t", "u", "g", "h", "j"], "u": ["y", "i", "h", "j", "k"],
    "i": ["u", "o", "j", "k", "l"], "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z"], "s": ["a", "w", "e", "d", "z", "x"],
    "d": ["s", "e", "r", "f", "x", "c"], "f": ["d", "r", "t", "g", "c", "v"],
    "g": ["f", "t", "y", "h", "v", "b"], "h": ["g", "y", "u", "j", "b", "n"],
    "j": ["h", "u", "i", "k", "n", "m"], "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "z": ["a", "s", "x"], "x": ["z", "s", "d", "c"], "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"], "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"], "m": ["n", "j", "k"],
}

VIET_MOBILE_ADJACENCY: dict[str, list[str]] = {
    "ă": ["a", "â"], "â": ["a", "ă"], "đ": ["d"],
    "ê": ["e"], "ô": ["o", "ơ"], "ơ": ["o", "ô"],
    "ư": ["u"],
}


def fat_finger(text: str, error_rate: float = 0.15) -> str:
    result = []
    for char in text:
        lower = char.lower()
        if random.random() < error_rate:
            candidates = VIET_MOBILE_ADJACENCY.get(lower) or QWERTY_ADJACENCY.get(lower)
            if candidates:
                replacement = random.choice(candidates)
                result.append(replacement.upper() if char.isupper() else replacement)
                continue
        result.append(char)
    return "".join(result)


__all__ = ["QWERTY_ADJACENCY", "VIET_MOBILE_ADJACENCY", "fat_finger"]
