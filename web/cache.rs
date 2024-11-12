use crate::types::CACHED_SESSION;
use ort::{Environment, Session};
use parking_lot::RwLockUpgradableReadGuard;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct CachedSession {
    pub session: Arc<Session>,
    pub last_used: Instant,
}

pub fn get_or_create_session(
    environment: &Arc<Environment>,
    model_path: &str,
) -> Result<Arc<Session>, Box<dyn std::error::Error>> {
    const CACHE_TIMEOUT: Duration = Duration::from_secs(3600);

    let read_guard = CACHED_SESSION.upgradable_read();

    if let Some(cached) = read_guard.as_ref() {
        if cached.last_used.elapsed() < CACHE_TIMEOUT {
            return Ok(cached.session.clone());
        }
    }

    let mut write_guard = RwLockUpgradableReadGuard::upgrade(read_guard);
    let new_session = Arc::new(
        ort::SessionBuilder::new(environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(i16::try_from(num_cpus::get()).unwrap_or(1))?
            .with_inter_threads(i16::try_from(num_cpus::get()).unwrap_or(1))?
            .with_memory_pattern(true)?
            .with_model_from_file(model_path)?,
    );

    *write_guard = Some(CachedSession {
        session: new_session.clone(),
        last_used: Instant::now(),
    });

    Ok(new_session)
}
