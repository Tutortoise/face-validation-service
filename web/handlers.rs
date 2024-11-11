use crate::{
    detection::process_image,
    types::{AppState, ValidationResponse},
};
use actix_multipart::Multipart;
use actix_web::{post, web, Error, HttpResponse};
use futures_util::TryStreamExt;

#[post("/validate-face")]
pub async fn validate_face(
    mut payload: Multipart,
    data: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    const MAX_SIZE: usize = 10 * 1024 * 1024; // 10MB limit
    const ALLOWED_MIME_TYPES: [&str; 3] = ["image/jpeg", "image/png", "image/webp"];

    while let Some(mut field) = payload.try_next().await? {
        if let Some(content_type) = field.content_type() {
            if !ALLOWED_MIME_TYPES.contains(&content_type.to_string().as_str()) {
                return Ok(HttpResponse::BadRequest().json(ValidationResponse {
                    is_valid: false,
                    face_count: 0,
                    message: "Invalid file type".to_string(),
                }));
            }
        }
        if let Some(cd) = field.content_disposition() {
            if cd.get_filename().is_some() {
                let mut bytes = Vec::with_capacity(MAX_SIZE / 2); // Initial reasonable capacity

                // Stream chunks with size limit
                while let Some(chunk) = field.try_next().await? {
                    if bytes.len() + chunk.len() > MAX_SIZE {
                        return Ok(HttpResponse::BadRequest().json(ValidationResponse {
                            is_valid: false,
                            face_count: 0,
                            message: "Image too large (max 10MB)".to_string(),
                        }));
                    }
                    bytes.extend_from_slice(&chunk);
                }

                // Check if we received any data
                if bytes.is_empty() {
                    return Ok(HttpResponse::BadRequest().json(ValidationResponse {
                        is_valid: false,
                        face_count: 0,
                        message: "Empty image file".to_string(),
                    }));
                }

                // Process image in a blocking thread
                let result =
                    web::block(move || image::load_from_memory(&bytes).map_err(|e| e.to_string()))
                        .await?;

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

                // Create a new session for this request
                let session = match ort::SessionBuilder::new(&data.environment)
                    .and_then(|builder| builder.with_model_from_file(&data.model_path))
                {
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

                // Process image and get face boxes
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

                        return Ok(HttpResponse::Ok().json(ValidationResponse {
                            is_valid,
                            face_count,
                            message,
                        }));
                    }
                    Err(e) => {
                        return Ok(
                            HttpResponse::InternalServerError().json(ValidationResponse {
                                is_valid: false,
                                face_count: 0,
                                message: format!("Error processing image: {}", e),
                            }),
                        );
                    }
                }
            }
        }
    }

    Ok(HttpResponse::BadRequest().json(ValidationResponse {
        is_valid: false,
        face_count: 0,
        message: "No image provided".to_string(),
    }))
}
