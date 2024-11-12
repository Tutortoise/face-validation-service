use crate::{
    cache::get_or_create_session,
    detection::process_image,
    types::{ApiResponse, AppState, ErrorCode, ErrorResponse, ValidationResponse},
};
use actix_multipart::Multipart;
use actix_web::{post, web, Error, HttpRequest, HttpResponse};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures_util::{StreamExt, TryStreamExt};

fn create_error_response(code: ErrorCode, message: &str, details: Option<&str>) -> ApiResponse {
    ApiResponse::Error(ErrorResponse {
        code,
        message: message.to_string(),
        details: details.map(String::from),
    })
}

#[post("/validate-face")]
pub async fn validate_face(
    req: HttpRequest,
    payload: web::Payload,
    data: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    const MAX_SIZE: usize = 10 * 1024 * 1024; // 10MB limit
    const ALLOWED_MIME_TYPES: [&str; 4] =
        ["image/jpeg", "image/png", "image/webp", "application/json"];

    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|ct| ct.to_str().ok())
        .unwrap_or("");

    let bytes = if content_type.starts_with("multipart/form-data") {
        process_multipart(content_type, payload, &ALLOWED_MIME_TYPES, MAX_SIZE).await?
    } else if content_type.starts_with("application/json") {
        process_json_payload(payload, MAX_SIZE).await?
    } else if ALLOWED_MIME_TYPES.contains(&content_type) {
        process_raw_file(payload, MAX_SIZE).await?
    } else {
        return Ok(
            HttpResponse::BadRequest().json(ApiResponse::Error(ErrorResponse {
                code: ErrorCode::InvalidContentType,
                message: "Unsupported content type".to_string(),
                details: Some("Use multipart/form-data, JSON, or direct image upload".to_string()),
            })),
        );
    };

    process_image_bytes(bytes, data).await
}

async fn process_json_payload(
    mut payload: web::Payload,
    max_size: usize,
) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::with_capacity(max_size / 2);

    while let Some(chunk) = payload.next().await {
        let chunk = chunk?;
        if bytes.len() + chunk.len() > max_size {
            return Err(actix_web::error::ErrorBadRequest(create_error_response(
                ErrorCode::FileTooLarge,
                "File too large",
                Some("Maximum file size is 10MB"),
            )));
        }
        bytes.extend_from_slice(&chunk);
    }

    // Parse JSON
    let json: serde_json::Value = serde_json::from_slice(&bytes)?;

    if let Some(base64_str) = json.get("image").and_then(|v| v.as_str()) {
        match STANDARD.decode(base64_str) {
            Ok(decoded) => Ok(decoded),
            Err(_) => Err(actix_web::error::ErrorBadRequest(create_error_response(
                ErrorCode::InvalidImageFormat,
                "Invalid base64 image data",
                None,
            ))),
        }
    } else {
        Err(actix_web::error::ErrorBadRequest(create_error_response(
            ErrorCode::NoFileProvided,
            "No image data in JSON",
            Some("Request must include base64 encoded image data"),
        )))
    }
}

async fn process_multipart(
    content_type: &str,
    payload: web::Payload,
    allowed_mime_types: &[&str],
    max_size: usize,
) -> Result<Vec<u8>, actix_web::Error> {
    let mut headers = actix_web::http::header::HeaderMap::new();
    if let Ok(header_value) = content_type.parse() {
        headers.insert(actix_web::http::header::CONTENT_TYPE, header_value);
    } else {
        return Err(actix_web::error::ErrorBadRequest(create_error_response(
            ErrorCode::InvalidContentType,
            "Invalid content-type header",
            None,
        )));
    }

    let mut multipart = Multipart::new(&headers, payload);

    if let Some(mut field) = multipart.try_next().await? {
        if let Some(content_type) = field.content_type() {
            if !allowed_mime_types.contains(&content_type.to_string().as_str()) {
                return Err(actix_web::error::ErrorBadRequest(create_error_response(
                    ErrorCode::UnsupportedFileType,
                    "Invalid file type",
                    Some("Only JPEG, PNG and WebP are supported"),
                )));
            }
        }

        let mut bytes = Vec::with_capacity(max_size / 2);
        while let Some(chunk) = field.try_next().await? {
            if bytes.len() + chunk.len() > max_size {
                return Err(actix_web::error::ErrorBadRequest(create_error_response(
                    ErrorCode::FileTooLarge,
                    "File too large",
                    Some("Maximum file size is 10MB"),
                )));
            }
            bytes.extend_from_slice(&chunk);
        }

        if !bytes.is_empty() {
            return Ok(bytes);
        }
    }

    Err(actix_web::error::ErrorBadRequest(create_error_response(
        ErrorCode::NoFileProvided,
        "No file provided",
        Some("Request must include an image file"),
    )))
}

async fn process_raw_file(mut payload: web::Payload, max_size: usize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::with_capacity(max_size / 2);
    while let Some(chunk) = payload.next().await {
        let chunk = chunk?;
        if bytes.len() + chunk.len() > max_size {
            return Err(actix_web::error::ErrorBadRequest(create_error_response(
                ErrorCode::FileTooLarge,
                "File too large",
                Some("Maximum file size is 10MB"),
            )));
        }
        bytes.extend_from_slice(&chunk);
    }

    if bytes.is_empty() {
        return Err(actix_web::error::ErrorBadRequest(create_error_response(
            ErrorCode::NoFileProvided,
            "Empty file",
            Some("Request must include an image file"),
        )));
    }

    Ok(bytes)
}

async fn process_image_bytes(
    bytes: Vec<u8>,
    data: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let session = match get_or_create_session(&data.environment, &data.model_path) {
        Ok(session) => session,
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(ApiResponse::Error(
                ErrorResponse {
                    code: ErrorCode::InternalError,
                    message: "Failed to initialize face detection".to_string(),
                    details: Some(e.to_string()),
                },
            )));
        }
    };

    let img = match web::block(move || image::load_from_memory(&bytes)).await? {
        Ok(img) => img,
        Err(e) => {
            return Ok(
                HttpResponse::BadRequest().json(ApiResponse::Error(ErrorResponse {
                    code: ErrorCode::InvalidImageFormat,
                    message: "Failed to decode image".to_string(),
                    details: Some(e.to_string()),
                })),
            );
        }
    };

    let result = process_image(img.clone(), session).await;
    drop(img);

    match result {
        Ok(boxes) => {
            let face_count = boxes.len();
            let (is_valid, message) = match face_count {
                0 => (false, "No faces detected".to_string()),
                1 => (true, "Valid single face detected".to_string()),
                n => (false, format!("Multiple faces detected: {}", n)),
            };

            Ok(
                HttpResponse::Ok().json(ApiResponse::Success(ValidationResponse {
                    is_valid,
                    face_count,
                    message,
                })),
            )
        }
        Err(e) => Ok(
            HttpResponse::InternalServerError().json(ApiResponse::Error(ErrorResponse {
                code: ErrorCode::ProcessingError,
                message: "Failed to process image".to_string(),
                details: Some(e.to_string()),
            })),
        ),
    }
}
