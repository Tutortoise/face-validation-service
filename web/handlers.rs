use crate::{
    cache::get_or_create_session,
    detection::process_image,
    types::{AppState, ValidationResponse},
};
use actix_multipart::Multipart;
use actix_web::{post, web, Error, HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};

#[post("/validate-face")]
pub async fn validate_face(
    req: HttpRequest,
    payload: web::Payload,
    data: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    const MAX_SIZE: usize = 10 * 1024 * 1024; // 10MB limit
    const ALLOWED_MIME_TYPES: [&str; 3] = ["image/jpeg", "image/png", "image/webp"];

    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|ct| ct.to_str().ok())
        .unwrap_or("");

    let bytes = if content_type.starts_with("multipart/form-data") {
        process_multipart(content_type, payload, &ALLOWED_MIME_TYPES, MAX_SIZE).await?
    } else if ALLOWED_MIME_TYPES.contains(&content_type) {
        process_raw_file(payload, MAX_SIZE).await?
    } else {
        return Ok(HttpResponse::BadRequest().json(ValidationResponse {
            is_valid: false,
            face_count: 0,
            message: "Unsupported content type. Use multipart/form-data or direct image upload"
                .to_string(),
        }));
    };

    process_image_bytes(bytes, data).await
}

async fn process_multipart(
    content_type: &str,
    payload: web::Payload,
    allowed_mime_types: &[&str],
    max_size: usize,
) -> Result<Vec<u8>, Error> {
    let mut headers = actix_web::http::header::HeaderMap::new();
    if let Ok(header_value) = content_type.parse() {
        headers.insert(actix_web::http::header::CONTENT_TYPE, header_value);
    } else {
        return Err(actix_web::error::ErrorBadRequest(
            "Invalid content-type header",
        ));
    }

    let mut multipart = Multipart::new(&headers, payload);

    if let Some(mut field) = multipart.try_next().await? {
        if let Some(content_type) = field.content_type() {
            if !allowed_mime_types.contains(&content_type.to_string().as_str()) {
                return Err(actix_web::error::ErrorBadRequest(
                    "Invalid file type. Only JPEG, PNG and WebP are supported",
                ));
            }
        }

        let mut bytes = Vec::with_capacity(max_size / 2);
        while let Some(chunk) = field.try_next().await? {
            if bytes.len() + chunk.len() > max_size {
                return Err(actix_web::error::ErrorPayloadTooLarge(
                    "File too large (max 10MB)",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }

        if !bytes.is_empty() {
            return Ok(bytes);
        }
    }

    Err(actix_web::error::ErrorBadRequest("No file provided"))
}

async fn process_raw_file(mut payload: web::Payload, max_size: usize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::with_capacity(max_size / 2);
    while let Some(chunk) = payload.next().await {
        let chunk = chunk?;
        if bytes.len() + chunk.len() > max_size {
            return Err(actix_web::error::ErrorPayloadTooLarge(
                "File too large (max 10MB)",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }

    if bytes.is_empty() {
        return Err(actix_web::error::ErrorBadRequest("Empty file"));
    }

    Ok(bytes)
}

async fn process_image_bytes(
    bytes: Vec<u8>,
    data: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let result =
        web::block(move || image::load_from_memory(&bytes).map_err(|e| e.to_string())).await?;

    let img = match result {
        Ok(img) => img,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().json(ValidationResponse {
                is_valid: false,
                face_count: 0,
                message: format!("Invalid image format: {}", e),
            }));
        }
    };

    let session = match get_or_create_session(&data.environment, &data.model_path) {
        Ok(session) => session,
        Err(e) => {
            return Ok(
                HttpResponse::InternalServerError().json(ValidationResponse {
                    is_valid: false,
                    face_count: 0,
                    message: format!("Failed to create session: {}", e),
                }),
            );
        }
    };

    match process_image(img, &session).await {
        Ok(boxes) => {
            let face_count = boxes.len();
            let is_valid = face_count == 1;
            let message = if is_valid {
                "Valid single face detected".to_string()
            } else if face_count == 0 {
                "No faces detected".to_string()
            } else {
                format!("Multiple faces detected: {}", face_count)
            };

            Ok(HttpResponse::Ok().json(ValidationResponse {
                is_valid,
                face_count,
                message,
            }))
        }
        Err(e) => Ok(
            HttpResponse::InternalServerError().json(ValidationResponse {
                is_valid: false,
                face_count: 0,
                message: format!("Error processing image: {}", e),
            }),
        ),
    }
}
