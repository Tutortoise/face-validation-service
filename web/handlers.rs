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
    while let Some(mut field) = payload.try_next().await? {
        if let Some(cd) = field.content_disposition() {
            if cd.get_filename().is_some() {
                let mut bytes = Vec::new();
                while let Some(chunk) = field.try_next().await? {
                    bytes.extend_from_slice(&chunk);
                }

                // Convert bytes to image
                let img = match image::load_from_memory(&bytes) {
                    Ok(img) => img,
                    Err(_) => {
                        return Ok(HttpResponse::BadRequest().json(ValidationResponse {
                            is_valid: false,
                            face_count: 0,
                            message: "Invalid image format".to_string(),
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
