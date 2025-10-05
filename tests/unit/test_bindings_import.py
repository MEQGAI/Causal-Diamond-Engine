import pytest


def test_bindings_importable() -> None:
    fm_bindings = pytest.importorskip(
        "fm_bindings", reason="bindings wheel not available"
    )
    assert fm_bindings is not None
