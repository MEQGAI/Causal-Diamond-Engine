"""Lightweight content filters used during dataset streaming."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable, Iterable, Mapping, Optional

logger = __import__("logging").getLogger(__name__)


def build_filter(specs: Iterable[Any]) -> Callable[[str, Mapping[str, Any]], bool]:
    """Return a predicate that enforces the provided filter specifications.

    Parameters
    ----------
    specs:
        An iterable of filter descriptors. Each descriptor can be either a
        string (e.g., ``"utf8"``) or a mapping describing the filter and its
        parameters (e.g., ``{"max_len": 8192}``).
    """

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
                params: Any = None
            elif isinstance(spec, Mapping):
                if not spec:
                    continue
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
                allowed = {code.lower() for code in _as_iterable(params)}
                lang = _infer_language(text, meta)
                if lang not in allowed:
                    return False
            elif name in {"near_dedup", "dedupe"}:
                threshold = float(params.get("threshold", 1.0)) if isinstance(params, Mapping) else 1.0
                if self._is_duplicate(text, threshold):
                    return False
            elif name in {"regex_deny", "deny_regex"} and params:
                pattern = re.compile(str(params))
                if pattern.search(text):
                    return False
            elif name in {"code_only", "code_filter"}:
                if not _is_code_like(text, meta):
                    return False
        return True

    def _is_duplicate(self, text: str, threshold: float) -> bool:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if digest in self._seen_hashes:
            return True
        # Very small threshold simply behaves like exact hash deduplication.
        self._seen_hashes.add(digest)
        return False


def _as_iterable(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
        return True
    except UnicodeEncodeError:  # pragma: no cover - defensive path
        return False


def _infer_language(text: str, meta: Mapping[str, Any]) -> str:
    if "lang" in meta:
        return str(meta["lang"]).lower()
    if "language" in meta:
        return str(meta["language"]).lower()
    letters = sum(ch.isalpha() for ch in text)
    non_ascii = sum(ord(ch) >= 128 for ch in text)
    if letters == 0:
        return ""  # unknown
    ratio = non_ascii / max(1, letters)
    return "en" if ratio < 0.2 else "multi"


def _is_code_like(text: str, meta: Mapping[str, Any]) -> bool:
    if "language" in meta:
        lang = str(meta["language"]).lower()
        return lang not in {"en", "english"}
    keyword_hits = sum(token in text for token in ("def ", "class ", "if (", "for ("))
    return keyword_hits >= 1


__all__ = ["build_filter"]
