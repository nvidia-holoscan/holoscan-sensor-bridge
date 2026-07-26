#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-process C++ library MDX emitted by ``fern docs md generate``."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PARAMFIELD_TYPE = re.compile(
    r'(<ParamField\b[^>]*?\btype=")([^"]*)(")',
    flags=re.DOTALL,
)
_AMPERSAND_ESCAPE = re.compile(
    r"&(?!amp;|lt;|gt;|quot;|apos;|#(?:[0-9]{1,7}|[xX][0-9A-Fa-f]{1,6});)"
)
_DEFAULT_STRING_LITERAL = re.compile(r'default=""([^"]*)""')
_UNFIXED_CPP_STRING_DEFAULT = re.compile(r'\bdefault=""')
_DESCRIPTION_LINE = re.compile(r'^(description:\s*")(.*)("\s*)$')
_FENCE_BLOCK = re.compile(r"^```[^\n]*\n[\s\S]*?^```\s*$", re.MULTILINE)


def fix_line(line: str) -> str:
    def replace_type(match: re.Match[str]) -> str:
        prefix, value, suffix = match.groups()
        return prefix + _AMPERSAND_ESCAPE.sub("&amp;", value) + suffix

    return PARAMFIELD_TYPE.sub(replace_type, line)


def _fix_description_line(line: str) -> str:
    raw = line.rstrip("\r\n")
    ending = line[len(raw) :]
    match = _DESCRIPTION_LINE.match(raw)
    if not match:
        return line
    prefix, body, suffix = match.groups()
    body = (
        body.replace("&lt;", "\x00lt\x00")
        .replace("&gt;", "\x00gt\x00")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\x00lt\x00", "&lt;")
        .replace("\x00gt\x00", "&gt;")
    )
    return prefix + body + suffix + ending


def _fix_html_comments(text: str) -> str:
    def replace_comment(match: re.Match[str]) -> str:
        body = match.group(1).replace("*/", "* /")
        return "" if "{" in body or "}" in body else "{/* " + body + " */}"

    chunks: list[str] = []
    position = 0
    for fence in _FENCE_BLOCK.finditer(text):
        chunks.append(
            re.sub(
                r"<!--([\s\S]*?)-->",
                replace_comment,
                text[position : fence.start()],
            )
        )
        chunks.append(fence.group(0))
        position = fence.end()
    chunks.append(re.sub(r"<!--([\s\S]*?)-->", replace_comment, text[position:]))
    return "".join(chunks)


def fix_file_content(text: str) -> str:
    text = _fix_html_comments(text)
    text = re.sub(r"<br\s*/?\s*>", "<br />", text, flags=re.IGNORECASE)
    text = _DEFAULT_STRING_LITERAL.sub(
        lambda match: "default='\"" + match.group(1) + "\"'",
        text,
    )
    lines = [_fix_description_line(line) for line in text.splitlines(keepends=True)]
    return "".join(fix_line(line) for line in lines)


def remove_library_overviews(root: Path) -> int:
    """Drop Fern's per-library overview.mdx so it does not appear in the sidebar."""
    removed = 0
    for path in sorted(root.rglob("overview.mdx")):
        path.unlink()
        removed += 1
        print(f"fix-generated-library-mdx: removed {path.relative_to(root)}")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "user_guide" / "fern" / "generated",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: generated MDX directory does not exist: {root}", file=sys.stderr)
        return 1

    remove_library_overviews(root)

    changed = 0
    errors: list[str] = []
    for path in sorted(root.rglob("*.mdx")):
        text = path.read_text(encoding="utf-8")
        fixed = fix_file_content(text)
        if fixed != text:
            path.write_text(fixed, encoding="utf-8")
            changed += 1
        for line_number, line in enumerate(fixed.splitlines(), 1):
            if _UNFIXED_CPP_STRING_DEFAULT.search(line):
                errors.append(f"{path.relative_to(root)}:{line_number}: {line.strip()}")

    print(f"fix-generated-library-mdx: updated {changed} file(s) under {root}")
    if errors:
        print("error: generated MDX still contains invalid C++ string defaults:", file=sys.stderr)
        for error in errors[:10]:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
