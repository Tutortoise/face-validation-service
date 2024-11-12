use crate::{
    clustering::cluster_boxes,
    types::{Detection, ProcessingError, CONF_THRESHOLD, INPUT_SIZE},
};
use image::DynamicImage;
use lazy_static::lazy_static;
use lru::LruCache;
use nalgebra::{Vector2, Vector4};
use ndarray::{Array, ArrayView1, CowArray};
use ort::{Session, Value};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::{sync::Arc, time::Duration};
use tokio::time::timeout;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const PROCESSING_TIMEOUT: Duration = Duration::from_secs(10);
const RETRY_ATTEMPTS: u32 = 3;
const RETRY_DELAY_MS: u64 = 100;

lazy_static! {
    pub(crate) static ref INPUT_BUFFER_CACHE: Mutex<LruCache<usize, Vec<f32>>> =
        Mutex::new(LruCache::new(std::num::NonZeroUsize::new(5).unwrap()));
}

pub fn cleanup_input_buffer_cache() {
    INPUT_BUFFER_CACHE.lock().clear();
}

pub fn cleanup_old_buffers() {
    let mut cache = INPUT_BUFFER_CACHE.lock();
    cache.clear();
}

pub async fn process_image(
    image: DynamicImage,
    session: Arc<Session>, // Change the parameter type to Arc<Session>
) -> Result<Vec<[i32; 4]>, ProcessingError> {
    let mut last_error = None;

    for attempt in 1..=RETRY_ATTEMPTS {
        let image_clone = image.clone();
        let session_clone = Arc::clone(&session); // Clone the Arc, not the session directly

        let processing =
            tokio::spawn(async move { process_image_internal(image_clone, &session_clone) });

        match timeout(PROCESSING_TIMEOUT, processing).await {
            Ok(Ok(Ok(boxes))) => return Ok(boxes),
            Ok(Ok(Err(e))) => {
                last_error = Some(e);
                if attempt < RETRY_ATTEMPTS {
                    tokio::time::sleep(Duration::from_millis(
                        (RETRY_DELAY_MS as u64) * (attempt as u64),
                    ))
                    .await;
                    continue;
                }
            }
            Ok(Err(e)) => {
                last_error = Some(ProcessingError::Internal(e.to_string()));
            }
            Err(_) => {
                last_error = Some(ProcessingError::Timeout);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| ProcessingError::Internal("Unknown error".to_string())))
}

fn process_image_internal(
    image: DynamicImage,
    session: &Session,
) -> Result<Vec<[i32; 4]>, ProcessingError> {
    let original_width = image.width();
    let original_height = image.height();

    // Convert image to RGB
    let rgb_image = image.to_rgb8();

    // Resize image
    let resized = image::imageops::resize(
        &rgb_image,
        INPUT_SIZE.0,
        INPUT_SIZE.1,
        image::imageops::FilterType::Triangle,
    );

    // Prepare input buffer
    let input_data = prepare_input_buffer(&resized)?;

    // Run inference
    let predictions = run_inference(session, input_data)?;

    let mut detections = process_predictions(predictions, original_width, original_height)?;

    Ok(cluster_boxes(&mut detections))
}

fn prepare_input_buffer(
    resized: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Result<Vec<f32>, ProcessingError> {
    let buffer_size = (INPUT_SIZE.0 * INPUT_SIZE.1 * 3) as usize;

    // Get or create buffer from cache
    let mut input_data = {
        let mut cache = INPUT_BUFFER_CACHE.lock();
        cache
            .get(&buffer_size)
            .cloned()
            .unwrap_or_else(|| Vec::with_capacity(buffer_size))
    };
    input_data.clear();

    // Process channels with error recovery
    for c in 0..3 {
        match process_channel_safely(resized, c) {
            Ok(channel_data) => input_data.extend(channel_data),
            Err(_) => {
                input_data.extend(process_channel_fallback(resized, c));
            }
        }
    }

    // Cache the buffer for reuse
    {
        let mut cache = INPUT_BUFFER_CACHE.lock();
        cache.put(buffer_size, input_data.clone());
    }

    Ok(input_data)
}

fn process_channel_safely(
    resized: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    channel: usize,
) -> Result<Vec<f32>, ProcessingError> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return Ok(unsafe { process_channel_simd(resized.as_raw(), channel) });
    }

    Ok(process_channel_fallback(resized, channel))
}

fn process_channel_fallback(
    resized: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    channel: usize,
) -> Vec<f32> {
    let pixels = resized.as_raw();
    let mut result = Vec::with_capacity((INPUT_SIZE.0 * INPUT_SIZE.1) as usize);

    for i in (channel..pixels.len()).step_by(3) {
        result.push(pixels[i] as f32 / 255.0);
    }

    result
}

fn run_inference(
    session: &Session,
    input_data: Vec<f32>,
) -> Result<ndarray::Array2<f32>, ProcessingError> {
    let shape = [1, 3, INPUT_SIZE.0 as usize, INPUT_SIZE.1 as usize];

    // Create array from input data
    let array = Array::from_shape_vec(shape, input_data)
        .map_err(|e| ProcessingError::Internal(format!("Failed to create input array: {}", e)))?;

    // Prepare input tensor
    let array = array.as_standard_layout().to_owned();
    let cow_array = CowArray::from(array);
    let cow_array_dyn = cow_array.into_dyn();

    let input_tensor = Value::from_array(session.allocator(), &cow_array_dyn).map_err(|e| {
        ProcessingError::InferenceError(format!("Failed to create input tensor: {}", e))
    })?;

    // Run inference
    let outputs = session
        .run(vec![input_tensor])
        .map_err(|e| ProcessingError::InferenceError(format!("Model inference failed: {}", e)))?;

    // Process output tensor
    let output_tensor = outputs[0].try_extract::<f32>().map_err(|e| {
        ProcessingError::InferenceError(format!("Failed to extract output tensor: {}", e))
    })?;

    let output_view = output_tensor.view();

    // Reshape predictions
    output_view
        .to_owned()
        .into_shape((1, 5, 8400))
        .map_err(|e| ProcessingError::Internal(format!("Failed to reshape output: {}", e)))?
        .permuted_axes([2, 1, 0])
        .as_standard_layout()
        .to_owned()
        .into_shape((8400, 5))
        .map_err(|e| ProcessingError::Internal(format!("Failed to reshape predictions: {}", e)))
}

fn process_predictions(
    predictions: ndarray::Array2<f32>,
    original_width: u32,
    original_height: u32,
) -> Result<Vec<Detection>, ProcessingError> {
    let detections: Vec<Detection> = predictions
        .axis_iter(ndarray::Axis(0))
        .par_bridge()
        .filter_map(|prediction| {
            let confidence = prediction[4];
            if confidence >= CONF_THRESHOLD {
                Some(create_detection(
                    prediction,
                    original_width,
                    original_height,
                    confidence,
                ))
            } else {
                None
            }
        })
        .collect();

    let mut detections = detections;
    detections.par_sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(detections)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn process_channel_simd(pixels: &[u8], channel: usize) -> Vec<f32> {
    let len = pixels.len();
    let mut result: Vec<f32> = Vec::with_capacity(len / 3);
    let mut i = channel;

    // Process 16 pixels at a time using two AVX2 registers
    while i + 48 <= len {
        let mut values: [f32; 16] = [0.0; 16];
        for j in 0..16 {
            values[j] = pixels[i + j * 3] as f32;
        }

        // Process first 8 pixels
        let pixels_f32_1 = _mm256_loadu_ps(values.as_ptr());
        let normalized_1 = _mm256_div_ps(pixels_f32_1, _mm256_set1_ps(255.0));

        // Process next 8 pixels
        let pixels_f32_2 = _mm256_loadu_ps(values.as_ptr().add(8));
        let normalized_2 = _mm256_div_ps(pixels_f32_2, _mm256_set1_ps(255.0));

        // Store results
        _mm256_storeu_ps(result.as_mut_ptr().add(result.len()), normalized_1);
        _mm256_storeu_ps(result.as_mut_ptr().add(result.len() + 8), normalized_2);

        result.set_len(result.len() + 16);
        i += 48; // Move forward 16 pixels * 3 channels
    }

    // Handle remaining pixels with original 8-pixel SIMD
    while i + 24 <= len {
        let mut values: [f32; 8] = [0.0; 8];
        for j in 0..8 {
            values[j] = pixels[i + j * 3] as f32;
        }

        let pixels_f32 = _mm256_loadu_ps(values.as_ptr());
        let normalized = _mm256_div_ps(pixels_f32, _mm256_set1_ps(255.0));
        _mm256_storeu_ps(result.as_mut_ptr().add(result.len()), normalized);

        result.set_len(result.len() + 8);
        i += 24;
    }

    // Handle remaining pixels
    while i < len {
        result.push(pixels[i] as f32 / 255.0);
        i += 3;
    }

    result
}

#[inline(always)]
fn create_detection(
    prediction: ArrayView1<f32>,
    original_width: u32,
    original_height: u32,
    confidence: f32,
) -> Detection {
    let pred_vec = Vector4::new(prediction[0], prediction[1], prediction[2], prediction[3]);

    let input_size = Vector4::new(
        INPUT_SIZE.0 as f32,
        INPUT_SIZE.1 as f32,
        INPUT_SIZE.0 as f32,
        INPUT_SIZE.1 as f32,
    );

    let abs_coords = pred_vec.component_mul(&input_size);
    let [abs_x_center, abs_y_center, abs_width, abs_height] =
        [abs_coords[0], abs_coords[1], abs_coords[2], abs_coords[3]];

    let half_sizes = Vector2::new(abs_width / 2.0, abs_height / 2.0);
    let center = Vector2::new(abs_x_center, abs_y_center);

    let corners_min = center - half_sizes;
    let corners_max = center + half_sizes;

    let scale = Vector2::new(
        original_width as f32 / INPUT_SIZE.0 as f32,
        original_height as f32 / INPUT_SIZE.1 as f32,
    );

    let scaled_min = (corners_min.component_mul(&scale)).map(|x| x.round() as i32);
    let scaled_max = (corners_max.component_mul(&scale)).map(|x| x.round() as i32);

    let bbox = [
        scaled_min.x.max(0),
        scaled_min.y.max(0),
        scaled_max.x.min(original_width as i32),
        scaled_max.y.min(original_height as i32),
    ];

    Detection { bbox, confidence }
}
