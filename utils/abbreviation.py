import re

ABBREVIATION_MAP: dict[str, str] = {
    "không": "k",
    "mọi người": "mn",
    "bình thường": "bt",
    "được": "đc",
    "biết": "bít",
    "với": "vs",
    "nhưng": "nhg",
    "vâng": "vg",
    "thôi": "th",
    "tại sao": "ts",
    "bao giờ": "bg",
    "ông": "og",
    "bà": "b",
    "anh": "a",
    "chị": "c",
    "em": "e",
    "tôi": "t",
    "mình": "mk",
    "chúng mình": "cm",
    "chúng ta": "ct",
    "thế thôi": "tt",
    "ví dụ": "vd",
    "điện thoại": "đt",
    "chồng": "ck",
    "vợ": "vk",
    "hay không": "hk",
    "rồi": "r",
    "luôn": "lun",
    "nhé": "nhe",
    "nha": "na",
    "thật": "tht",
    "thật sự": "ts",
    "tất nhiên": "tn",
    "có thể": "có thể",
    "ngày": "ng",
    "giờ": "g",
    "phút": "p",
}


def abbreviate(text: str) -> str:
    result = text
    for full, abbr in sorted(ABBREVIATION_MAP.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(full), re.IGNORECASE)
        result = pattern.sub(abbr, result)
    return result


__all__ = ["ABBREVIATION_MAP", "abbreviate"]
