import random
from typing import Callable

VIET_ALPHABET = list(
    "aăâbcdđeêghiklmnoôơpqrstuưvxy"
    "àảãáạằẳẵắặầẩẫấậ"
    "èẻẽéẹềểễếệ"
    "ìỉĩíị"
    "òỏõóọồổỗốộờởỡớợ"
    "ùủũúụừửữứự"
    "ỳỷỹýỵ"
)


def _insert(text: str) -> str:
    pos = random.randint(0, len(text))
    char = random.choice(VIET_ALPHABET)
    return text[:pos] + char + text[pos:]


def _delete(text: str) -> str:
    if len(text) <= 1:
        return text
    pos = random.randint(0, len(text) - 1)
    return text[:pos] + text[pos + 1:]


def _substitute(text: str) -> str:
    if not text:
        return text
    pos = random.randint(0, len(text) - 1)
    char = random.choice(VIET_ALPHABET)
    return text[:pos] + char + text[pos + 1:]


def _transpose(text: str) -> str:
    if len(text) < 2:
        return text
    pos = random.randint(0, len(text) - 2)
    lst = list(text)
    lst[pos], lst[pos + 1] = lst[pos + 1], lst[pos]
    return "".join(lst)


EDIT_OPS: list[Callable[[str], str]] = [_insert, _delete, _substitute, _transpose]


def edit_distance_error(text: str, num_edits: int = 1) -> str:
    result = text
    for _ in range(num_edits):
        op = random.choice(EDIT_OPS)
        result = op(result)
    return result


__all__ = ["VIET_ALPHABET", "EDIT_OPS", "edit_distance_error"]
