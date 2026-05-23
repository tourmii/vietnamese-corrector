import random

from tone_utils import VOWEL_TONES

TELEX_RAW: dict[str, str] = {
    "à": "af", "ả": "ar", "ã": "ax", "á": "as", "ạ": "aj",
    "ầ": "aaf", "ẩ": "aar", "ẫ": "aax", "ấ": "aas", "ậ": "aaj",
    "ằ": "awf", "ẳ": "awr", "ẵ": "awx", "ắ": "aws", "ặ": "awj",
    "ề": "eef", "ể": "eer", "ễ": "eex", "ế": "ees", "ệ": "eej",
    "ì": "if", "ỉ": "ir", "ĩ": "ix", "í": "is", "ị": "ij",
    "ồ": "oof", "ổ": "oor", "ỗ": "oox", "ố": "oos", "ộ": "ooj",
    "ờ": "owf", "ở": "owr", "ỡ": "owx", "ớ": "ows", "ợ": "owj",
    "ù": "uf", "ủ": "ur", "ũ": "ux", "ú": "us", "ụ": "uj",
    "ừ": "uwf", "ử": "uwr", "ữ": "uwx", "ứ": "uws", "ự": "uwj",
    "ỳ": "yf", "ỷ": "yr", "ỹ": "yx", "ý": "ys", "ỵ": "yj",
    "ă": "aw", "â": "aa", "ê": "ee", "ô": "oo", "ơ": "ow", "ư": "uw", "đ": "dd",
}

TELEX_DROP_TONE: dict[str, str] = {}
for accented, (base, tone) in VOWEL_TONES.items():
    if tone != 0:
        TELEX_DROP_TONE[accented] = base


def telex_error(text: str, mode: str = "random") -> str:
    if mode == "random":
        mode = random.choice(["raw", "drop_tone"])

    result = []
    for char in text:
        lower = char.lower()
        if mode == "raw" and lower in TELEX_RAW:
            raw = TELEX_RAW[lower]
            result.append(raw.upper() if char.isupper() else raw)
        elif mode == "drop_tone" and lower in TELEX_DROP_TONE:
            base = TELEX_DROP_TONE[lower]
            result.append(base.upper() if char.isupper() else base)
        else:
            result.append(char)
    return "".join(result)


__all__ = ["TELEX_RAW", "TELEX_DROP_TONE", "telex_error"]
