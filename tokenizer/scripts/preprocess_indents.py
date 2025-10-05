#!/usr/bin/env python3
"""Convert leading indentation into explicit tokens."""
from __future__ import annotations

import re
import sys

for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    match = re.match(r"^[ \t]+", line)
    if not match:
        print(line)
        continue

    prefix = match.group(0)
    rest = line[len(prefix) :]
    tokens: list[str] = []

    tabs = prefix.count("\t")
    tokens.extend(["<|tab|>"] * tabs)

    spaces_only = prefix.replace("\t", "")
    count = len(spaces_only)
    if count:
        tokens.extend(["<|indent_4|>"] * (count // 4))
        remainder = count % 4
        if remainder:
            tokens.append(f"<|indent_{remainder}|>")

    print("".join(tokens) + rest)
