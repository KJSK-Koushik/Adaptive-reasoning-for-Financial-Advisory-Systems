"""Rendering helpers shared by the dataset adapters.

FinQA and TAT-QA store tables as lists of lists and reports as lists of sentences.
The reasoning model sees plain text, so everything is flattened here - in one place,
so every source is presented to the model in a consistent format.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Collapse whitespace and strip. FinQA text is full of ragged spacing."""
    return _WS.sub(" ", (text or "").replace(" ", " ")).strip()


def render_table(rows: list[list[str]], max_rows: int = 25) -> str:
    """Render a list-of-lists table as a pipe-delimited block.

    Markdown-style pipes are used because reasoning models handle them well and they
    cost far fewer tokens than JSON. Rows beyond ``max_rows`` are dropped with a note
    rather than silently, so a truncated table is never mistaken for a complete one.
    """
    if not rows:
        return ""

    truncated = len(rows) > max_rows
    body = rows[:max_rows]

    lines = []
    for i, row in enumerate(body):
        cells = [clean(str(c)) for c in row]
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append(" | ".join("---" for _ in cells))

    if truncated:
        lines.append(f"... ({len(rows) - max_rows} further rows omitted)")
    return "\n".join(lines)


def truncate_context(text: str, max_chars: int) -> str:
    """Trim context to a character budget, cutting at a sentence boundary if possible.

    Keeps the **end** of the narrative text, since in financial filings the sentences
    nearest the table are usually the relevant ones.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    tail = text[-max_chars:]
    # Prefer to start at a clean sentence boundary.
    for marker in (". ", "\n"):
        idx = tail.find(marker)
        if 0 <= idx < max_chars // 4:
            return tail[idx + len(marker):].strip()
    return tail.strip()


def build_context(narrative: str, table: str, max_chars: int) -> str:
    """Assemble a context block, giving the table priority over the prose.

    The table is where the numbers live, so it is never truncated to make room for
    narrative - the narrative is trimmed instead.
    """
    table = table.strip()
    narrative = clean(narrative)

    remaining = max_chars - len(table) - 2
    if remaining <= 0:
        return table

    narrative = truncate_context(narrative, remaining)
    return f"{narrative}\n\n{table}".strip() if narrative else table


def format_number(value: float) -> str:
    """Render a gold answer compactly without scientific notation or trailing zeros."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
