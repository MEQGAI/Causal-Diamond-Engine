use engine::CausalDiamondEngine;
pub use pyo3::prelude::*;

#[pyclass]
struct PyCausalDiamondEngine {
    inner: CausalDiamondEngine,
}

#[pymethods]
impl PyCausalDiamondEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: CausalDiamondEngine::new(),
        }
    }

    fn step(&mut self, input: &str, budget: f32) -> PyResult<String> {
        self.inner
            .step(input, budget)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))
    }
}

#[pyfunction]
fn banner() -> String {
    engine::banner()
}

#[pymodule]
fn ledger_python(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCausalDiamondEngine>()?;
    m.add_function(wrap_pyfunction!(banner, m)?)?;
    Ok(())
}
