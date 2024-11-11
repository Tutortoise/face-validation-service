use crate::{
    clustering::cluster_boxes,
    types::{Detection, CONF_THRESHOLD, INPUT_SIZE},
};
use image::DynamicImage;
use ndarray::{Array, ArrayView1, CowArray};
use ort::{Session, Value};
use rayon::prelude::*;

pub async fn process_image(
    image: DynamicImage,
    session: &Session,
) -> Result<Vec<[i32; 4]>, Box<dyn std::error::Error>> {
    // Store original dimensions
    let original_width = image.width();
    let original_height = image.height();

    let rgb_image = image.to_rgb8();

    let resized = image::imageops::resize(
        &rgb_image,
        INPUT_SIZE.0,
        INPUT_SIZE.1,
        image::imageops::FilterType::Triangle,
    );

    let mut input_data = Vec::with_capacity((INPUT_SIZE.0 * INPUT_SIZE.1 * 3) as usize);
    for c in 0..3 {
        let channel_data: Vec<f32> = (0..INPUT_SIZE.1)
            .into_par_iter()
            .flat_map(|y| {
                let row_pixels: Vec<f32> = (0..INPUT_SIZE.0)
                    .map(|x| resized.get_pixel(x, y)[c] as f32 / 255.0)
                    .collect();
                row_pixels
            })
            .collect();
        input_data.extend(channel_data);
    }

    // Create input tensor
    let shape = [1, 3, INPUT_SIZE.0 as usize, INPUT_SIZE.1 as usize];
    let array = Array::from_shape_vec(shape, input_data)?;

    // Ensure proper memory layout
    let array = array.as_standard_layout().to_owned();
    let cow_array = CowArray::from(array);
    let cow_array_dyn = cow_array.into_dyn();

    // Run inference
    let input_tensor = Value::from_array(session.allocator(), &cow_array_dyn)?;
    let outputs = session.run(vec![input_tensor])?;

    // Process output with careful handling of dimensions
    let output_tensor = outputs[0].try_extract::<f32>()?;
    let output_view = output_tensor.view();

    let predictions = output_view
        .to_owned()
        .into_shape((1, 5, 8400))?
        .permuted_axes([2, 1, 0])
        .as_standard_layout()
        .to_owned()
        .into_shape((8400, 5))?;

    let detections: Vec<Detection> = predictions
        .axis_iter(ndarray::Axis(0))
        .par_bridge()
        .filter_map(|prediction| {
            let confidence = prediction[4];
            if confidence >= CONF_THRESHOLD {
                // Process detection in parallel
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

    // Sort after parallel processing
    let mut detections = detections;
    detections.par_sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Ok(cluster_boxes(&mut detections))
}

#[inline(always)]
fn create_detection(
    prediction: ArrayView1<f32>,
    original_width: u32,
    original_height: u32,
    confidence: f32,
) -> Detection {
    let x_center = prediction[0];
    let y_center = prediction[1];
    let width = prediction[2];
    let height = prediction[3];

    // Convert normalized coordinates to absolute pixel coordinates
    let abs_x_center = x_center * INPUT_SIZE.0 as f32;
    let abs_y_center = y_center * INPUT_SIZE.1 as f32;
    let abs_width = width * INPUT_SIZE.0 as f32;
    let abs_height = height * INPUT_SIZE.1 as f32;

    // Calculate corner coordinates in input size space
    let x1 = (abs_x_center - abs_width / 2.0).round() as i32;
    let y1 = (abs_y_center - abs_height / 2.0).round() as i32;
    let x2 = (abs_x_center + abs_width / 2.0).round() as i32;
    let y2 = (abs_y_center + abs_height / 2.0).round() as i32;

    // Scale to original image size
    let scale_x = original_width as f32 / INPUT_SIZE.0 as f32;
    let scale_y = original_height as f32 / INPUT_SIZE.1 as f32;

    let x1 = (x1 as f32 * scale_x).round() as i32;
    let y1 = (y1 as f32 * scale_y).round() as i32;
    let x2 = (x2 as f32 * scale_x).round() as i32;
    let y2 = (y2 as f32 * scale_y).round() as i32;

    // Ensure coordinates are within image bounds
    let bbox = [
        x1.max(0),
        y1.max(0),
        x2.min(original_width as i32),
        y2.min(original_height as i32),
    ];

    Detection { bbox, confidence }
}
