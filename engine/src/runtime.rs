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
    modal::{self, ModalOutcome, ModalState},
    stability::{self, StabilityOutcome},
    telemetry,
};

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

pub trait CausalDiamondEngine {
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

impl CausalDiamondEngine for Engine {
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
