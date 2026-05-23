import unicodedata


def no_diacritic(text: str) -> str:
	normalized = unicodedata.normalize("NFD", text)
	stripped = "".join(
		char for char in normalized if unicodedata.category(char) != "Mn"
	)
	return stripped.replace("đ", "d").replace("Đ", "D")


__all__ = ["no_diacritic"]
