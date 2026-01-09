import re

def strip_gutenberg_text(text: str) -> str:
    """
    Removes Project Gutenberg boilerplate and trims text
    to start at the first real chapter.
    """

    # 1. Remove Gutenberg header
    start_match = re.search(
        r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if start_match:
        text = text[start_match.end():]

    # 2. Remove Gutenberg footer
    end_match = re.search(
        r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if end_match:
        text = text[: end_match.start()]

    # 3. Trim everything before first chapter
    chapter_match = re.search(
        r"\n\s*(chapter\s+[ivxlcdm0-9]+\.?|chapter\s+one|part\s+i)\s*\n",
        text,
        re.IGNORECASE,
    )

    if chapter_match:
        text = text[chapter_match.start():]

    return text.strip()
