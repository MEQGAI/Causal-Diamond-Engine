import numpy as np
import pytest

engine_py = pytest.importorskip(
    "engine_py", reason="PyO3 bindings not built; run `maturin develop`"
)


def _toy_batch():
    lambda_vals = np.linspace(-0.1, 0.1, 5, dtype=np.float64)
    theta = np.stack([np.linspace(0.0, 0.2, 5, dtype=np.float64)])
    shear = np.zeros_like(theta)
    d_area = np.ones_like(theta) * 4.0
    sqrt_h = np.array([1.0], dtype=np.float64)
    return {
        "lambda_": lambda_vals,
        "theta": theta,
        "shear2": shear,
        "dA": d_area,
        "sqrt_h": sqrt_h,
        "meta": {"affine_half_width": 0.1, "radius": 1.0, "curvature": 0.0},
    }


def test_engine_submit_and_join():
    eng = engine_py.Engine({"precision": "f64", "backend": "ndarray"})
    eng.prepare()
    handle = eng.submit_batch(_toy_batch())
    report = eng.join(handle)
    assert isinstance(report, dict)
    assert "qfc_pass_rate" in report


def test_engine_step_summary():
    eng = engine_py.Engine(None)
    payload = {"step": 1, "delta": 0.1}
    out = eng.step(payload, 1.0)
    assert isinstance(out, dict)
    assert "summary" in out


def test_engine_version():
    ver = engine_py.Engine.version()
    assert isinstance(ver, str)
    assert ver
