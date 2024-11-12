mod cache;
mod clustering;
mod detection;
mod handlers;
mod types;

use actix_web::{middleware, web, App, HttpServer};
use handlers::validate_face;
use ort::Environment;
use std::sync::Arc;
use std::time::Duration;
use tokio::time; // Add this
use types::AppState;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let environment = Arc::new(
        Environment::builder()
            .with_name("face-validation")
            .with_log_level(ort::LoggingLevel::Warning)
            .build()
            .unwrap(),
    );

    let model_path = "models/yolo11n_9ir_640_hface.onnx".to_string();

    let server = HttpServer::new(move || {
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
    .keep_alive(Duration::from_secs(30))
    .client_request_timeout(Duration::from_secs(60))
    .bind("127.0.0.1:8080")?;

    let cleanup_interval = Duration::from_secs(3600); // Every hour
    let cleanup_task = tokio::spawn(async move {
        let mut interval = time::interval(cleanup_interval);
        loop {
            interval.tick().await;
            cache::cleanup_expired_sessions();
            detection::cleanup_old_buffers();
        }
    });

    // Improved shutdown handling
    let (tx, rx) = tokio::sync::oneshot::channel();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!("Received shutdown signal");
        let _ = tx.send(());
    });

    let server_handle = server.run();

    tokio::select! {
        result = server_handle => {
            println!("Server stopped: {:?}", result);
        }
        _ = rx => {
            println!("Shutting down...");
            detection::cleanup_input_buffer_cache();
            cache::cleanup_session_cache();
            cleanup_task.abort();
            time::sleep(Duration::from_secs(1)).await;
        }
    }

    Ok(())
}
