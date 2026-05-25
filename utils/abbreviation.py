import re

ABBREVIATION_MAP: dict[str, str] = {
    "không": "k",
    "mọi người": "mn",
    "bình thường": "bt",
    "được": "đc",
    "biết": "bt",
    "với": "vs",
    "nhưng": "nhg",
    "vâng": "vg",
    "thôi": "th",
    "tại sao": "tsao",
    "bao giờ": "bh",
    "ông": "og",
    "bạn": "b",
    "anh": "a",
    "chị": "c",
    "em": "e",
    "tôi": "t",
    "mình": "mk",
    "chúng mình": "cm",
    "chúng ta": "cta",
    "thế thôi": "tt",
    "ví dụ": "vd",
    "điện thoại": "đth",
    "chồng": "ck",
    "vợ": "vk",
    "hay không": "hk",
    "rồi": "r",
    "luôn": "lun",
    "nhé": "nhe",
    "nha": "na",
    "thật": "tht",
    "thật sự": "ts",
    "tin nhắn": "tn",
    "ngày": "ng",
    "giờ": "h",
    "phút": "p",
    "các thứ": "ct"
}


def abbreviate(text: str) -> str:
    result = text
    for full, abbr in sorted(ABBREVIATION_MAP.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(full), re.IGNORECASE)
        result = pattern.sub(abbr, result)
    return result


__all__ = ["ABBREVIATION_MAP", "abbreviate"]
