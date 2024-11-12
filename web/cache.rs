use dashmap::DashMap;
use lazy_static::lazy_static;
use ort::{Environment, Session};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct CachedSession {
    pub session: Arc<Session>,
    pub last_used: Instant,
}

lazy_static! {
    pub(crate) static ref SESSION_CACHE: DashMap<String, CachedSession> =
        DashMap::with_capacity(NonZeroUsize::new(10).unwrap().get());
}

pub fn cleanup_session_cache() {
    SESSION_CACHE.clear();
}

pub fn cleanup_expired_sessions() {
    const MAX_AGE: Duration = Duration::from_secs(3600 * 4); // 4 hours
    SESSION_CACHE.retain(|_, cached| cached.last_used.elapsed() < MAX_AGE);
}

pub fn get_or_create_session(
    environment: &Arc<Environment>,
    model_path: &str,
) -> Result<Arc<Session>, Box<dyn std::error::Error>> {
    const CACHE_TIMEOUT: Duration = Duration::from_secs(3600);

    if let Some(cached) = SESSION_CACHE.get(model_path) {
        if cached.last_used.elapsed() < CACHE_TIMEOUT {
            return Ok(cached.session.clone());
        }
        SESSION_CACHE.remove(model_path);
    }

    let new_session = Arc::new(
        ort::SessionBuilder::new(environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(i16::try_from(num_cpus::get().max(1)).unwrap_or(1))?
            .with_inter_threads(i16::try_from(num_cpus::get().max(1)).unwrap_or(1))?
            .with_memory_pattern(true)?
            .with_model_from_file(model_path)?,
    );

    SESSION_CACHE.insert(
        model_path.to_string(),
        CachedSession {
            session: new_session.clone(),
            last_used: Instant::now(),
        },
    );

    Ok(new_session)
}
