use anyhow::Result;
use axum::{routing::get, Json, Router};
use engine::CausalDiamondEngine;
use serde::Serialize;
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

#[derive(Serialize)]
struct StepResponse {
    output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let engine = Arc::new(Mutex::new(CausalDiamondEngine::new()));
    let app = Router::new().route(
        "/healthz",
        get({
            let engine = Arc::clone(&engine);
            move || async move {
                let mut engine = engine.lock().await;
                let result = engine
                    .step("health", 1.0)
                    .unwrap_or_else(|_| "error".into());
                Json(StepResponse { output: result })
            }
        }),
    );

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("ledger server listening at http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
