mod clustering;
mod detection;
mod handlers;
mod types;

use actix_web::{middleware, web, App, HttpServer};
use handlers::validate_face;
use ort::Environment;
use std::sync::Arc;
use types::AppState;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let environment = Arc::new(
        Environment::builder()
            .with_name("face-validation")
            .build()
            .unwrap(),
    );

    let model_path = "models/yolo11n_9ir_640_hface.onnx".to_string();

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState {
                environment: environment.clone(),
                model_path: model_path.clone(),
            }))
            .wrap(middleware::Logger::new("%r %s %D ms"))
            .wrap(middleware::Compress::default())
            .wrap(middleware::NormalizePath::trim())
            .service(validate_face)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
