use std::collections::HashMap;
use std::time::{Duration, Instant};

use engine::config::{
    Concurrency, EngineConfig, Features, FrwConfig, IoConfig, LedgerConfig, Precision,
    StabilityConfig, TensorBackend,
};
use engine::geometry::{Diamond, DiamondMeta, NullGen, Screen};
use engine::runtime::{CausalDiamondEngine, DiamondBatch, EngineReport, JobHandle};
use numpy::{PyArray1, PyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyTimeoutError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use uuid::Uuid;

create_exception!(engine_py, EngineError, pyo3::exceptions::PyException);

#[pyclass(module = "engine_py", name = "Engine")]
struct PyEngine {
    inner: CausalDiamondEngine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(py: Python<'_>, config: Option<&PyAny>) -> PyResult<Self> {
        let cfg = parse_engine_config(py, config)?;
        Ok(Self {
            inner: CausalDiamondEngine::with_config(cfg),
        })
    }

    fn prepare(&mut self) -> PyResult<()> {
        self.inner
            .prepare()
            .map_err(|err| EngineError::new_err(err.to_string()))
    }

    fn configure(&mut self, py: Python<'_>, config: &PyAny) -> PyResult<()> {
        let cfg = parse_engine_config(py, Some(config))?;
        self.inner.configure(cfg);
        Ok(())
    }

    fn submit_batch(&mut self, py: Python<'_>, batch: &PyAny) -> PyResult<String> {
        let batch = build_diamond_batch(py, batch)?;
        let handle = self
            .inner
            .submit_batch(batch)
            .map_err(|err| EngineError::new_err(err.to_string()))?;
        Ok(handle.to_string())
    }

    #[args(timeout_ms = "None")]
    fn join(&mut self, handle: &str, timeout_ms: Option<u64>) -> PyResult<PyObject> {
        let parsed = JobHandle::from_str(handle)
            .map_err(|err| EngineError::new_err(format!("invalid job handle: {err}")))?;
        let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms));
        loop {
            match self.inner.join(parsed) {
                Ok(report) => return Python::with_gil(|py| Ok(report_to_dict(py, report)?)),
                Err(err) if err.to_string().contains("unknown job") => {
                    if let Some(dl) = deadline {
                        if Instant::now() >= dl {
                            return Err(PyTimeoutError::new_err("join timed out"));
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(err) => return Err(EngineError::new_err(err.to_string())),
            }
        }
    }

    fn step(&mut self, py: Python<'_>, input: &PyAny, budget: f32) -> PyResult<PyObject> {
        let json_mod = py.import("json")?;
        let serialized = if let Ok(s) = input.extract::<String>() {
            s
        } else {
            json_mod
                .call_method1("dumps", (input,))?
                .extract::<String>()?
        };
        let result = self
            .inner
            .step(&serialized, budget)
            .map_err(|err| EngineError::new_err(err.to_string()))?;
        // Try to parse as JSON for convenience.
        if let Ok(value) = json_mod.call_method1("loads", (result.as_str(),)) {
            return Ok(value.into());
        }
        let dict = PyDict::new(py);
        dict.set_item("summary", result)?;
        Ok(dict.into())
    }

    fn evaluate_batch(&mut self, py: Python<'_>, batch: &PyAny) -> PyResult<PyObject> {
        let batch = build_diamond_batch(py, batch)?;
        let report = self
            .inner
            .evaluate_batch(batch)
            .map_err(|err| EngineError::new_err(err.to_string()))?;
        Python::with_gil(|py| report_to_dict(py, report))
    }

    fn shutdown(&mut self) -> PyResult<()> {
        self.inner
            .shutdown()
            .map_err(|err| EngineError::new_err(err.to_string()))
    }

    #[staticmethod]
    fn version() -> &'static str {
        CausalDiamondEngine::version()
    }
}

#[pyfunction]
fn banner() -> String {
    engine::banner()
}

#[pymodule]
fn engine_py(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(banner, m)?)?;
    m.add("EngineError", py.get_type::<EngineError>())?;
    Ok(())
}

fn parse_engine_config(py: Python<'_>, obj: Option<&PyAny>) -> PyResult<PyEngineConfig> {
    let mut cfg = EngineConfig::default();
    let Some(any) = obj else {
        return Ok(PyEngineConfig { inner: cfg });
    };
    if any.is_none() {
        return Ok(PyEngineConfig { inner: cfg });
    }
    let dict = any
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("EngineConfig must be a mapping with string keys"))?;

    if let Some(val) = dict.get_item("precision") {
        let s = val.extract::<&str>()?.to_lowercase();
        cfg.precision = match s.as_str() {
            "f32" => Precision::F32,
            "f64" => Precision::F64,
            other => return Err(PyValueError::new_err(format!("invalid precision: {other}"))),
        };
    }

    if let Some(val) = dict.get_item("backend") {
        let s = val.extract::<&str>()?.to_lowercase();
        cfg.tensor_backend = match s.as_str() {
            "ndarray" => TensorBackend::NdArray,
            "nalgebra" => TensorBackend::Nalgebra,
            "tch" | "torch" => TensorBackend::TchTorch,
            "cpusimd" | "cpu_simd" => TensorBackend::CpuSimd,
            other => return Err(PyValueError::new_err(format!("invalid backend: {other}"))),
        };
    }

    if let Some(val) = dict.get_item("concurrency") {
        let conc = val
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("concurrency must be a dict such as {'rayon': 8}"))?;
        if let Some(workers) = conc.get_item("rayon") {
            let workers = workers.extract::<usize>()?;
            cfg.concurrency = Concurrency::Rayon { workers };
        } else {
            cfg.concurrency = Concurrency::Single;
        }
    }

    if let Some(val) = dict.get_item("stability") {
        let stab = val
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("stability must be a dict"))?;
        let mut stability = StabilityConfig::default();
        if let Some(qfc) = stab.get_item("qfc_tolerance") {
            stability.qfc_tolerance = qfc.extract::<f64>()?;
        }
        if let Some(hmin) = stab.get_item("hessian_min") {
            stability.hessian_min = hmin.extract::<f64>()?;
        }
        cfg.stability = stability;
    }

    if let Some(val) = dict.get_item("io") {
        let io = val
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("io must be a dict"))?;
        let mut io_cfg = IoConfig::default();
        if let Some(d) = io.get_item("diamonds") {
            io_cfg.diamonds = Some(d.extract::<String>()?);
        }
        if let Some(s) = io.get_item("state") {
            io_cfg.state = Some(s.extract::<String>()?);
        }
        if let Some(out) = io.get_item("outputs") {
            io_cfg.outputs = Some(out.extract::<String>()?);
        }
        cfg.io = io_cfg;
    }

    if let Some(val) = dict.get_item("features") {
        let list = val
            .downcast::<PyList>()
            .map_err(|_| PyTypeError::new_err("features must be a list of strings"))?;
        let mut features = Features::default();
        for item in list.iter() {
            let flag = item.extract::<&str>()?.to_lowercase();
            match flag.as_str() {
                "wald" => features.wald = true,
                "gaussian_ledger" => features.gaussian_ledger = true,
                "pointer_ledger" => features.pointer_ledger = true,
                "frw_map" => features.frw_map = true,
                "gpu" => features.gpu = true,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported feature toggle: {other}"
                    )))
                }
            }
        }
        cfg.features = features;
    }

    if let Some(val) = dict.get_item("frw") {
        let map = val
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("frw must be a dict"))?;
        let alpha1 = map
            .get_item("alpha1")
            .map(|v| v.extract::<f64>())
            .transpose()?;
        let alpha2 = map
            .get_item("alpha2")
            .map(|v| v.extract::<f64>())
            .transpose()?;
        if let (Some(alpha1), Some(alpha2)) = (alpha1, alpha2) {
            let mut frw = FrwConfig {
                alpha1,
                alpha2,
                kappa: 0.2,
            };
            if let Some(kappa) = map.get_item("kappa") {
                frw.kappa = kappa.extract::<f64>()?;
            }
            cfg.frw = Some(frw);
        }
    }

    if let Some(val) = dict.get_item("ledger") {
        let map = val
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("ledger must be a dict"))?;
        if let Some(kappa) = map.get_item("kappa_bar") {
            cfg.ledger = Some(LedgerConfig {
                kappa_bar: kappa.extract::<f64>()?,
            });
        }
    }

    Ok(PyEngineConfig { inner: cfg })
}

fn build_diamond_batch(_py: Python<'_>, any: &PyAny) -> PyResult<DiamondBatch> {
    if let Ok(dict) = any.downcast::<PyDict>() {
        return batch_from_numpy(dict);
    }
    Err(PyTypeError::new_err(
        "DiamondBatch must be provided as a mapping",
    ))
}

fn batch_from_numpy(dict: &PyDict) -> PyResult<DiamondBatch> {
    let lambda = extract_vec1(dict, "lambda_")?;
    let theta = extract_vec2(dict, "theta")?;
    let shear2 = extract_vec2(dict, "shear2")?;
    let d_area = extract_vec2(dict, "dA")?;
    let sqrt_h = extract_vec1(dict, "sqrt_h")?;
    let generators = theta.len();

    let meta = dict
        .get_item("meta")
        .and_then(|v| v.downcast::<PyDict>().ok());

    let mut diamond_generators = Vec::with_capacity(generators);
    for idx in 0..generators {
        diamond_generators.push(NullGen {
            lambda: lambda.clone(),
            theta: theta[idx].clone(),
            shear2: shear2[idx].clone(),
            d_area: d_area[idx].clone(),
            stress_kk: None,
        });
    }

    let mut coords = Vec::new();
    if let Some(arr) = dict
        .get_item("coords")
        .and_then(|v| v.downcast::<PyArray2<f64>>().ok())
    {
        let view = arr.readonly();
        for row in view.as_array().outer_iter() {
            if row.len() >= 2 {
                coords.push([row[0], row[1]]);
            }
        }
    } else {
        for _ in 0..sqrt_h.len() {
            coords.push([0.0, 0.0]);
        }
    }

    let screen = Screen { coords, sqrt_h };

    let mut diamond_meta = DiamondMeta::default();
    if let Some(meta) = meta {
        if let Some(v) = meta.get_item("affine_half_width") {
            diamond_meta.affine_half_width = v.extract::<f64>()?;
        }
        if let Some(v) = meta.get_item("radius") {
            diamond_meta.radius = v.extract::<f64>()?;
        }
        if let Some(v) = meta.get_item("curvature") {
            diamond_meta.curvature = v.extract::<f64>()?;
        }
    }

    let id = meta
        .and_then(|m| m.get_item("id"))
        .map(|v| v.extract::<u64>())
        .transpose()?
        .unwrap_or_else(|| Uuid::new_v4().as_u128() as u64);

    let diamond = Diamond {
        id,
        leaf: screen,
        generators: diamond_generators,
        meta: diamond_meta,
    };

    Ok(DiamondBatch {
        diamonds: vec![diamond],
        states: HashMap::new(),
    })
}

fn extract_vec1(dict: &PyDict, key: &str) -> PyResult<Vec<f64>> {
    let arr = dict
        .get_item(key)
        .ok_or_else(|| PyKeyError::new_err(format!("missing key '{key}'")))?;
    if let Ok(array) = arr.downcast::<PyArray1<f64>>() {
        Ok(array.readonly().as_slice()?.to_vec())
    } else if let Ok(array) = arr.downcast::<PyArray1<f32>>() {
        Ok(array
            .readonly()
            .as_slice()?
            .iter()
            .map(|v| *v as f64)
            .collect())
    } else {
        Err(PyTypeError::new_err(format!(
            "{key} must be a 1D numpy array of floats"
        )))
    }
}

fn extract_vec2(dict: &PyDict, key: &str) -> PyResult<Vec<Vec<f64>>> {
    let arr = dict
        .get_item(key)
        .ok_or_else(|| PyKeyError::new_err(format!("missing key '{key}'")))?;
    if let Ok(array) = arr.downcast::<PyArray2<f64>>() {
        let view = array.readonly();
        let mut out = Vec::new();
        for row in view.as_array().outer_iter() {
            out.push(row.to_vec());
        }
        Ok(out)
    } else if let Ok(array) = arr.downcast::<PyArray2<f32>>() {
        let view = array.readonly();
        let mut out = Vec::new();
        for row in view.as_array().outer_iter() {
            out.push(row.iter().map(|v| *v as f64).collect());
        }
        Ok(out)
    } else {
        Err(PyTypeError::new_err(format!(
            "{key} must be a 2D numpy array of floats"
        )))
    }
}

fn report_to_dict(py: Python<'_>, report: EngineReport) -> PyResult<PyObject> {
    let total = report.diamonds.len() as f64;
    let mut theta_min = f64::INFINITY;
    let mut theta_max = f64::NEG_INFINITY;
    let mut hessian_min = f64::INFINITY;
    let mut qfc_passes = 0.0;
    let mut violations = 0_usize;
    let mut modal_tau = Vec::new();

    for diamond in &report.diamonds {
        if let Some(min_q) = diamond
            .stability
            .theta_quantum
            .iter()
            .cloned()
            .reduce(f64::min)
        {
            theta_min = theta_min.min(min_q);
        }
        if let Some(max_q) = diamond
            .stability
            .theta_quantum
            .iter()
            .cloned()
            .reduce(f64::max)
        {
            theta_max = theta_max.max(max_q);
        }
        hessian_min = hessian_min.min(diamond.stability.hessian_min);
        if diamond.stability.qfc_pass {
            qfc_passes += 1.0;
        }
        violations += diamond.stability.failures.len();
        modal_tau.push(diamond.modal.tau_modal);
    }

    let flags = PyDict::new(py);
    flags.set_item("qfc_violation", report.qfc_failures > 0)?;
    flags.set_item("modal_fallback", report.modal_fallbacks > 0)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "theta_q_min",
        if theta_min.is_finite() {
            theta_min
        } else {
            0.0
        },
    )?;
    dict.set_item(
        "theta_q_max",
        if theta_max.is_finite() {
            theta_max
        } else {
            0.0
        },
    )?;
    dict.set_item(
        "qfc_pass_rate",
        if total > 0.0 { qfc_passes / total } else { 1.0 },
    )?;
    dict.set_item(
        "hessian_min_eig",
        if hessian_min.is_finite() {
            hessian_min
        } else {
            0.0
        },
    )?;
    dict.set_item("violations", violations)?;
    dict.set_item("flags", flags)?;
    if !modal_tau.is_empty() {
        let mean: f64 = modal_tau.iter().sum::<f64>() / modal_tau.len() as f64;
        dict.set_item("modal_energy_density", mean)?;
    } else {
        dict.set_item("modal_energy_density", py.None())?;
    }
    dict.set_item("summary", report.summary())?;
    Ok(dict.into())
}
