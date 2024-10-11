import re
import unicodedata

from num2words import num2words


def truncate_text(text: str, max_bytes: int = 5000):
    """Truncate text to a maximum of max_bytes bytes."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def normalize_text(text: str, task: str = "transcription") -> str:
    """Normalize text for comparison."""
    # Convert to lowercase
    text = text.lower()

    # Unicode normalization (convert to standard form)
    text = unicodedata.normalize("NFKD", text)

    # Remove diacritical marks
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Replace numbers with their word equivalents
    def replace_number(match):
        number = match.group()
        try:
            return num2words(int(number))
        except ValueError:
            return num2words(float(number))

    text = re.sub(r"\b\d+(?:\.\d+)?\b", replace_number, text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text
