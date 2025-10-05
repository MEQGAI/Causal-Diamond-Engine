use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    config::{Concurrency, EngineConfig},
    entanglement::{self, StateSlice},
    errors::{EngineError, Result},
    geometry::{self, Diamond, GeometryOutcome},
    io::simple::StoredBatch,
    modal::{self, ModalOutcome, ModalState},
    stability::{self, StabilityOutcome},
    telemetry,
};
use serde_json::Value;

/// Bundle of state information for a diamond.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateBundle {
    pub entanglement: StateSlice,
    pub modal: ModalState,
}

/// Batch of diamonds processed together by the engine.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiamondBatch {
    pub diamonds: Vec<Diamond>,
    pub states: HashMap<u64, StateBundle>,
}

/// Individual diamond report collecting channel outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiamondReport {
    pub id: u64,
    pub geometry: GeometryOutcome,
    pub entanglement: entanglement::EntanglementOutcome,
    pub modal: ModalOutcome,
    pub stability: StabilityOutcome,
}

/// Aggregated report returned to the caller.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineReport {
    pub diamonds: Vec<DiamondReport>,
    pub qfc_failures: usize,
    pub modal_fallbacks: usize,
}

impl EngineReport {
    pub fn summary(&self) -> String {
        format!(
            "diamonds={}, qfc_failures={}, modal_fallbacks={}",
            self.diamonds.len(),
            self.qfc_failures,
            self.modal_fallbacks
        )
    }
}

/// Handle representing an in-flight or completed job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobHandle(Uuid);

pub trait EngineRuntime {
    fn prepare(&mut self, cfg: EngineConfig) -> Result<()>;
    fn submit(&self, batch: DiamondBatch) -> Result<JobHandle>;
    fn join(&self, handle: JobHandle) -> Result<EngineReport>;
    fn shutdown(&mut self) -> Result<()>;
}

/// Concrete engine implementation.
pub struct Engine {
    cfg: Option<EngineConfig>,
    pool: Option<Arc<ThreadPool>>,
    jobs: Arc<Mutex<HashMap<Uuid, EngineReport>>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            cfg: None,
            pool: None,
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Engine {
    pub fn new() -> Self {
        Self::default()
    }

    fn evaluate_batch(cfg: &EngineConfig, batch: DiamondBatch) -> Result<EngineReport> {
        let mut reports = Vec::with_capacity(batch.diamonds.len());
        let mut qfc_failures = 0;
        let mut modal_fallbacks = 0;

        for diamond in &batch.diamonds {
            let state = batch.states.get(&diamond.id).cloned().unwrap_or_default();
            let geom = geometry::evaluate_diamond(diamond, &cfg.features);
            let ent = entanglement::evaluate(diamond, &state.entanglement);
            let modal = match modal::evaluate(&state.modal) {
                Ok(outcome) => outcome,
                Err(err) => {
                    modal_fallbacks += 1;
                    tracing::warn!(target: "engine", diamond_id = diamond.id, "modal backend failed: {err}");
                    let mut fallback = ModalOutcome::default();
                    fallback.fallback = Some(err.to_string());
                    fallback
                }
            };
            let stability = stability::evaluate(diamond, &geom, &ent, &modal, &cfg.stability);
            if !stability.qfc_pass {
                qfc_failures += 1;
            }

            telemetry::record_geometry(&geom);
            telemetry::record_entanglement(&ent);
            telemetry::record_modal(&modal);
            telemetry::record_stability(&stability);

            reports.push(DiamondReport {
                id: diamond.id,
                geometry: geom,
                entanglement: ent,
                modal,
                stability,
            });
        }

        Ok(EngineReport {
            diamonds: reports,
            qfc_failures,
            modal_fallbacks,
        })
    }
}

impl EngineRuntime for Engine {
    fn prepare(&mut self, cfg: EngineConfig) -> Result<()> {
        if let Concurrency::Rayon { workers } = cfg.concurrency {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|err| EngineError::other(format!("failed to build rayon pool: {err}")))?;
            self.pool = Some(Arc::new(pool));
        }
        self.cfg = Some(cfg);
        Ok(())
    }

    fn submit(&self, batch: DiamondBatch) -> Result<JobHandle> {
        let cfg = self
            .cfg
            .as_ref()
            .ok_or_else(|| EngineError::other("engine not prepared"))?;
        let report = if let Some(pool) = &self.pool {
            let cfg = cfg.clone();
            pool.install(|| Engine::evaluate_batch(&cfg, batch))?
        } else {
            Engine::evaluate_batch(cfg, batch)?
        };

        let id = Uuid::new_v4();
        self.jobs.lock().insert(id, report);
        Ok(JobHandle(id))
    }

    fn join(&self, handle: JobHandle) -> Result<EngineReport> {
        self.jobs
            .lock()
            .remove(&handle.0)
            .ok_or_else(|| EngineError::other("unknown job handle"))
    }

    fn shutdown(&mut self) -> Result<()> {
        self.cfg = None;
        self.pool = None;
        self.jobs.lock().clear();
        Ok(())
    }
}

/// High-level handle exposed to external callers (CLI, Python bindings).
pub struct CausalDiamondEngine {
    inner: Engine,
    cfg: EngineConfig,
    prepared: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        entanglement::StateSlice,
        geometry::make_small_diamond,
        modal::ModalState,
    };
    use std::collections::HashMap;

    #[test]
    fn step_evaluates_serialised_batch() {
        let diamond = make_small_diamond(7, 1.0, 0.05);
        let mut ent = HashMap::new();
        ent.insert(diamond.id, StateSlice::default());
        let mut modal = HashMap::new();
        modal.insert(diamond.id, ModalState::default());
        let stored = StoredBatch {
            diamonds: vec![diamond],
            entanglement: ent,
            modal,
        };
        let payload = serde_json::to_string(&stored).unwrap();
        let mut engine = CausalDiamondEngine::new();
        let summary = engine.step(&payload, 1.0).expect("engine step to succeed");
        assert!(summary.contains("diamonds=1"), "summary missing diamonds count: {summary}");
    }

    #[test]
    fn step_records_telemetry_payload() {
        let payload = serde_json::json!({
            "step": 1,
            "delta": 0.5,
            "loss_mod": 0.1
        })
        .to_string();
        let mut engine = CausalDiamondEngine::new();
        let summary = engine.step(&payload, 0.75).expect("telemetry step");
        assert_eq!(summary, "ack");
    }
}

impl Default for CausalDiamondEngine {
    fn default() -> Self {
        Self {
            inner: Engine::new(),
            cfg: EngineConfig::default(),
            prepared: false,
        }
    }
}

impl CausalDiamondEngine {
    /// Construct a new engine handle with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a specific configuration.
    pub fn with_config(cfg: EngineConfig) -> Self {
        Self {
            cfg,
            ..Self::default()
        }
    }

    /// Update configuration and force a re-prepare on next use.
    pub fn configure(&mut self, cfg: EngineConfig) {
        self.cfg = cfg;
        self.prepared = false;
    }

    fn ensure_prepared(&mut self) -> Result<()> {
        if !self.prepared {
            self.inner.prepare(self.cfg.clone())?;
            self.prepared = true;
        }
        Ok(())
    }

    /// Submit a batch and return the evaluated report.
    pub fn evaluate_batch(&mut self, batch: DiamondBatch) -> Result<EngineReport> {
        self.ensure_prepared()?;
        let handle = self.inner.submit(batch)?;
        self.inner.join(handle)
    }

    /// Convenience step: accepts either a serialized diamond batch or a
    /// telemetry payload and returns a short status string.
    pub fn step(&mut self, payload: &str, budget: f32) -> Result<String> {
        if let Ok(stored) = serde_json::from_str::<StoredBatch>(payload) {
            let batch: DiamondBatch = stored.into();
            let report = self.evaluate_batch(batch)?;
            Ok(format!("budget={budget:.2}, {}", report.summary()))
        } else {
            self.record_telemetry(payload, budget);
            Ok("ack".to_string())
        }
    }

    fn record_telemetry(&self, payload: &str, budget: f32) {
        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            if let Some(delta) = value
                .get("delta")
                .and_then(|v| v.as_f64())
            {
                metrics::gauge!("engine.ledger.delta", delta);
            }
            if let Some(loss_mod) = value
                .get("loss_mod")
                .and_then(|v| v.as_f64())
            {
                metrics::gauge!("engine.ledger.loss_mod", loss_mod);
            }
            if let Some(step) = value
                .get("step")
                .and_then(|v| v.as_i64())
            {
                metrics::counter!("engine.ledger.steps", 1, "step" => step.to_string());
            }
        }
        metrics::gauge!("engine.ledger.budget", budget as f64);
        tracing::debug!(target = "engine", payload, %budget, "ingested telemetry payload");
    }
}
