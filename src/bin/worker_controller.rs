use std::{env, time::Duration};

use anyhow::Result;
use carabiner::instances;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let database_url = env::var("DATABASE_URL").unwrap_or_else(|_| {
        "postgres://mountaineer:mountaineer@localhost:5433/mountaineer_daemons".to_string()
    });
    tracing::info!("starting worker controller loop");
    instances::wait_for_instance_poll(&database_url, Duration::from_secs(1)).await?;
    Ok(())
}
