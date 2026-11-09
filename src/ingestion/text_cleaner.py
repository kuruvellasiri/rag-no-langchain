import re


def clean_text(text: str) -> str:

    # Normalize multiple spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove unnecessary spaces around lines
    text = "\n".join(
        line.strip() for line in text.splitlines()
    )

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()