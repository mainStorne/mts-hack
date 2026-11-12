import re

_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def language(text: str) -> str:
    """'Russian' or 'English' by the letters of the text; decided in code, as models kept English for Russian questions."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "Russian"
    return "Russian" if len(_CYRILLIC.findall(text)) * 2 >= len(letters) else "English"
