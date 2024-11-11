use ort::Environment;
use serde::Serialize;
use std::{fmt, sync::Arc};

// Constants
pub const INPUT_SIZE: (u32, u32) = (640, 640);
pub const CONF_THRESHOLD: f32 = 0.6;
pub const IOU_THRESHOLD: f32 = 0.45;
pub const DEFAULT_CLUSTER_SIZE: f64 = 50.0;

#[derive(Clone)]
pub struct Detection {
    pub bbox: [i32; 4],
    pub confidence: f32,
}

#[derive(Serialize)]
pub struct ValidationResponse {
    pub is_valid: bool,
    pub face_count: usize,
    pub message: String,
}

pub struct AppState {
    pub environment: Arc<Environment>,
    pub model_path: String,
}

#[derive(Debug)]
pub struct OrtErrorWrapper(pub String);

impl fmt::Display for OrtErrorWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ONNX Runtime error: {}", self.0)
    }
}

impl actix_web::ResponseError for OrtErrorWrapper {}

impl From<Box<dyn std::error::Error>> for OrtErrorWrapper {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        OrtErrorWrapper(err.to_string())
    }
}
