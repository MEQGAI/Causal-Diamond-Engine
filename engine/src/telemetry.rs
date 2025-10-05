use metrics::{counter, gauge};

use crate::{
    entanglement::EntanglementOutcome, geometry::GeometryOutcome, modal::ModalOutcome,
    stability::StabilityOutcome,
};

pub fn record_geometry(geometry: &GeometryOutcome) {
    gauge!("engine.geometry.delta_area", geometry.delta_area as f64);
    gauge!("engine.geometry.s_geo", geometry.s_geo as f64);
}

pub fn record_entanglement(ent: &EntanglementOutcome) {
    gauge!("engine.entanglement.delta_s", ent.delta_s_ent as f64);
}

pub fn record_modal(modal: &ModalOutcome) {
    gauge!("engine.modal.s_mod", modal.s_mod as f64);
    counter!("engine.modal.backend", 1, "backend" => format!("{:?}", modal.backend));
}

pub fn record_stability(stability: &StabilityOutcome) {
    counter!(
        "engine.stability.failures",
        stability.failures.len() as u64,
        "qfc" => stability.qfc_pass.to_string()
    );
    if let Some(max) = stability.dtheta_dlambda.iter().cloned().reduce(f64::max) {
        gauge!("engine.stability.max_dtheta", max);
    }
}
