import unicodedata


def remove_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt, chỉ giữ ký tự ASCII cơ bản."""
    replacements = {"Đ": "D", "đ": "d"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")
