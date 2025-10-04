"""Dataset catalog placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Curriculum:
    name: str
    description: str


_CATALOG: dict[str, Curriculum] = {
    "tool_reasoning": Curriculum(
        name="tool_reasoning",
        description="Synthetic curriculum for long-horizon tool use.",
    ),
    "stability_probing": Curriculum(
        name="stability_probing",
        description="Stress-tests modal ledger null-stability logic.",
    ),
}


def get_curriculum(name: str) -> Curriculum:
    try:
        return _CATALOG[name]
    except KeyError as err:
        raise KeyError(f"Unknown curriculum: {name}") from err
