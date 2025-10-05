"""Lightweight content filters used during dataset streaming."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Iterable, Mapping

logger = __import__("logging").getLogger(__name__)


def build_filter(specs: Iterable[Any]) -> Callable[[str, Mapping[str, Any]], bool]:
    chain = _FilterChain(specs)

    def _predicate(text: str, meta: Mapping[str, Any]) -> bool:
        return chain.apply(text, meta)

    return _predicate


class _FilterChain:
    def __init__(self, specs: Iterable[Any]) -> None:
        self.specs = list(specs)
        self._seen_hashes: set[str] = set()

    def apply(self, text: str, meta: Mapping[str, Any]) -> bool:
        for spec in self.specs:
            if isinstance(spec, str):
                name = spec
                params = None
            elif isinstance(spec, Mapping) and spec:
                name, params = next(iter(spec.items()))
            else:
                continue

            name = str(name).lower()
            if name in {"utf8", "utf-8"}:
                if not _is_utf8(text):
                    return False
            elif name in {"max_len", "max_length"}:
                limit = int(params)
                if len(text) > limit:
                    return False
            elif name in {"min_len", "min_length"}:
                limit = int(params)
                if len(text) < limit:
                    return False
            elif name in {"lang_allow", "language_allow"}:
                allowed = {str(x).lower() for x in _as_iterable(params)}
                lang = _infer_language(text, meta)
                if lang not in allowed:
                    return False
            elif name in {"near_dedup", "dedupe"}:
                if self._is_duplicate(text):
                    return False
            elif name in {"regex_deny", "deny_regex"} and params:
                pattern = re.compile(str(params))
                if pattern.search(text):
                    return False
            elif name in {"code_only", "code_filter"}:
                if not _is_code_like(text, meta):
                    return False
        return True

    def _is_duplicate(self, text: str) -> bool:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if digest in self._seen_hashes:
            return True
        self._seen_hashes.add(digest)
        return False


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
        return True
    except UnicodeEncodeError:  # pragma: no cover
        return False


def _infer_language(text: str, meta: Mapping[str, Any]) -> str:
    if "lang" in meta:
        return str(meta["lang"]).lower()
    if "language" in meta:
        return str(meta["language"]).lower()
    letters = sum(ch.isalpha() for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)
    if letters == 0:
        return ""
    ratio = non_ascii / max(1, letters)
    return "en" if ratio < 0.2 else "multi"


def _is_code_like(text: str, meta: Mapping[str, Any]) -> bool:
    if "language" in meta:
        lang = str(meta["language"]).lower()
        return lang not in {"en", "english"}
    snippet = text.lstrip()
    return snippet.startswith("def ") or snippet.startswith("class ") or "::" in snippet


__all__ = ["build_filter"]
