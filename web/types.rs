use crate::cache::CachedSession;
use lazy_static::lazy_static;
use ort::Environment;
use parking_lot::RwLock;
use serde::Serialize;
use std::{fmt, sync::Arc};
use thiserror::Error;

lazy_static! {
    pub(crate) static ref CACHED_SESSION: RwLock<Option<CachedSession>> = RwLock::new(None);
}

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

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidContentType,
    FileTooLarge,
    NoFileProvided,
    InvalidImageFormat,
    UnsupportedFileType,
    ProcessingError,
    InternalError,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            self.message,
            self.details
                .as_ref()
                .map(|d| format!(": {}", d))
                .unwrap_or_default()
        )
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ApiResponse {
    Success(ValidationResponse),
    Error(ErrorResponse),
}

impl fmt::Display for ApiResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiResponse::Success(response) => write!(
                f,
                "Success: {} (face count: {})",
                response.message, response.face_count
            ),
            ApiResponse::Error(error) => write!(f, "Error: {}", error),
        }
    }
}

impl fmt::Display for ValidationResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (valid: {}, face count: {})",
            self.message, self.is_valid, self.face_count
        )
    }
}

#[derive(Debug, Error)]
pub enum ProcessingError {
    #[error("Image processing timed out")]
    Timeout,
    #[error("Failed to load image: {0}")]
    ImageLoadError(#[from] image::ImageError),
    #[error("Model inference error: {0}")]
    InferenceError(String),
    #[error("Internal error: {0}")]
    Internal(String),
}
