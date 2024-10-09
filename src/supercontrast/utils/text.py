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
    text = unicodedata.normalize("NFKC", text)

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

    if task == "translation":
        # Additional normalization steps for translation
        # (e.g., handling of diacritics might differ)
        pass

    return text


def language_name_to_code(language_name: str) -> str:
    """Convert language name to ISO 639-1 code."""
    language_map = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
    }
    return language_map.get(language_name, "en")  # Default to English if not found
