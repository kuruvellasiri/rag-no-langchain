
import re


# Recognize common bullet symbols and numbered list items.
BULLET_PATTERN = re.compile(
    r"^\s*[•●▪■◦‣⁃◼▫◾]\s*"
)


NUMBERED_PATTERN = re.compile(
    r"^\s*(?:\d+[\.\)]|[a-zA-Z][\.\)])\s+"
)


def is_bullet(line: str) -> bool:
    """Check whether a line starts with a bullet symbol."""
    return bool(BULLET_PATTERN.match(line))


def is_numbered_item(line: str) -> bool:
    """Check whether a line starts with a numbered list marker."""
    return bool(NUMBERED_PATTERN.match(line))


def join_wrapped_lines(lines: list[str]) -> str:
    """
    Join PDF lines that belong to the same paragraph.

    Also fixes words split by a hyphen at a line boundary.
    """
    result = ""

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if not result:
            result = line

        elif result.endswith("-"):
            # Example: trans- + fer becomes transfer
            result = result[:-1] + line

        else:
            result += " " + line

    return result


def clean_text(text: str) -> str:

    # Step 1: Normalize spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Step 2: Remove unnecessary spaces around lines.
    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    # Step 3: Remove empty lines from the beginning and end.
    while lines and not lines[0]:
        lines.pop(0)

    while lines and not lines[-1]:
        lines.pop()

    # Step 4: Group lines into paragraphs and list items.
    blocks = []
    current_lines = []
    current_type = None

    def save_current_block():
        nonlocal current_lines, current_type

        if not current_lines:
            return

        content = join_wrapped_lines(current_lines)

        if content:
            if current_type == "bullet":
                blocks.append("• " + content)
            elif current_type == "numbered":
                blocks.append(content)
            else:
                blocks.append(content)

        current_lines = []
        current_type = None

    # Step 5: Process each extracted PDF line.
    for line in lines:

        # Blank line usually indicates a paragraph boundary.
        if not line:
            save_current_block()
            continue

        # A new bullet point starts.
        if is_bullet(line):
            save_current_block()

            content = BULLET_PATTERN.sub("", line, count=1)

            current_lines = [content]
            current_type = "bullet"
            continue

        # A new numbered list item starts.
        if is_numbered_item(line):
            save_current_block()

            current_lines = [line]
            current_type = "numbered"
            continue

        # Continue the existing paragraph or list item.
        if current_type is None:
            current_type = "paragraph"

        current_lines.append(line)

    # Save the final block.
    save_current_block()

    # Step 6: Separate paragraphs and list items.
    return "\n\n".join(blocks).strip()
