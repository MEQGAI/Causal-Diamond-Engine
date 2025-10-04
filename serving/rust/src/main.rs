use axum::{routing::get, Router};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let app = Router::new().route("/health", get(healthcheck));

    // TODO: wire Axum handlers into engine runtime once refactor lands.
    let addr: SocketAddr = "127.0.0.1:8000".parse()?;
    tracing::info!("starting serving shim", %addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn healthcheck() -> &'static str {
    "ok"
}
