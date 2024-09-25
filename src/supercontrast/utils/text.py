def truncate_text(text: str, max_bytes: int = 5000):
    """Truncate text to a maximum of max_bytes bytes."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")
