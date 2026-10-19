"""Markdown tables for small experiment summaries."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


def _stringify_cell(value: Any, formatter: str | Callable[[Any], str] | None = None) -> str:
    if formatter is None:
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)
    if isinstance(formatter, str):
        return formatter.format(value)
    return formatter(value)


def format_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    headers: Mapping[str, str] | None = None,
    formats: Mapping[str, str | Callable[[Any], str]] | None = None,
    title: str | None = None,
) -> str:
    """Return a compact Markdown table.

    This intentionally avoids pandas.  It is for small experiment summaries that
    should be easy to read in terminals, Markdown files, and notebooks.
    """
    headers = headers or {}
    formats = formats or {}
    labels = [headers.get(col, col) for col in columns]
    body = [
        [_stringify_cell(row.get(col, ""), formats.get(col)) for col in columns]
        for row in rows
    ]
    widths = [
        max(len(labels[i]), *(len(row[i]) for row in body)) if body else len(labels[i])
        for i in range(len(columns))
    ]

    def line(values: Sequence[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    parts = []
    if title:
        parts.extend([title, ""])
    parts.append(line(labels))
    parts.append(sep)
    parts.extend(line(row) for row in body)
    return "\n".join(parts)
