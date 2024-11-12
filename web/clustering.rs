use crate::types::{Detection, DEFAULT_CLUSTER_SIZE, IOU_THRESHOLD};
use dbscan::cluster;
use rayon::prelude::*;

pub fn cluster_boxes(detections: &mut Vec<Detection>) -> Vec<[i32; 4]> {
    if detections.is_empty() {
        return Vec::new();
    }

    let mut points = Vec::with_capacity(detections.len());
    points.par_extend(detections.par_iter().map(|det| {
        vec![
            det.bbox[0] as f64,
            det.bbox[1] as f64,
            det.bbox[2] as f64,
            det.bbox[3] as f64,
        ]
    }));

    let eps = calculate_median_size(detections).max(DEFAULT_CLUSTER_SIZE) * 0.5;
    let min_points = if detections.len() > 3 { 2 } else { 1 };

    let clusters = cluster(eps, min_points, &points);
    if !clusters
        .iter()
        .all(|c| matches!(c, dbscan::Classification::Noise))
    {
        return process_clusters(detections, clusters);
    }
    process_clusters(detections, cluster(eps * 1.5, min_points, &points))
}

fn calculate_median_size(detections: &[Detection]) -> f64 {
    let mut sizes: Vec<f64> = detections
        .iter()
        .map(|det| {
            let width = (det.bbox[2] - det.bbox[0]) as f64;
            let height = (det.bbox[3] - det.bbox[1]) as f64;
            (width * height).sqrt()
        })
        .collect();

    sizes.sort_by(safe_f64_cmp);
    sizes
        .get(sizes.len() / 2)
        .copied()
        .unwrap_or(DEFAULT_CLUSTER_SIZE)
}

fn process_clusters(
    detections: &[Detection],
    clusters: Vec<dbscan::Classification>,
) -> Vec<[i32; 4]> {
    let mut final_boxes = Vec::new();
    let mut cluster_map: std::collections::HashMap<usize, Vec<[i32; 4]>> =
        std::collections::HashMap::new();

    for (idx, classification) in clusters.iter().enumerate() {
        if idx >= detections.len() {
            continue;
        }

        match classification {
            dbscan::Classification::Core(cluster_id) | dbscan::Classification::Edge(cluster_id) => {
                cluster_map
                    .entry(*cluster_id)
                    .or_default()
                    .push(detections[idx].bbox);
            }
            dbscan::Classification::Noise => {
                handle_noise_point(&mut final_boxes, &mut cluster_map, detections[idx].bbox);
            }
        }
    }

    finalize_clusters(&mut final_boxes, cluster_map, detections)
}

fn handle_noise_point(
    final_boxes: &mut Vec<[i32; 4]>,
    cluster_map: &mut std::collections::HashMap<usize, Vec<[i32; 4]>>,
    bbox: [i32; 4],
) {
    let mut merged = false;
    for boxes in cluster_map.values_mut() {
        if boxes.iter().any(|existing_box| {
            let iou = calculate_iou(&bbox, existing_box);
            iou.is_finite() && iou > IOU_THRESHOLD
        }) {
            boxes.push(bbox);
            merged = true;
            break;
        }
    }

    if !merged {
        final_boxes.push(bbox);
    }
}

fn finalize_clusters(
    final_boxes: &mut Vec<[i32; 4]>,
    cluster_map: std::collections::HashMap<usize, Vec<[i32; 4]>>,
    detections: &[Detection],
) -> Vec<[i32; 4]> {
    for boxes in cluster_map.values() {
        if !boxes.is_empty() {
            final_boxes.push(merge_boxes(boxes));
        }
    }

    if final_boxes.is_empty() {
        let mut seen = std::collections::HashSet::new();
        final_boxes.extend(
            detections
                .iter()
                .filter(|det| seen.insert(det.bbox))
                .map(|det| det.bbox),
        );
    }

    final_boxes.to_vec()
}

pub fn safe_f64_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

#[inline(always)]
pub fn merge_boxes(boxes: &[[i32; 4]]) -> [i32; 4] {
    if boxes.is_empty() {
        return [0, 0, 0, 0];
    }

    let mut min_x = boxes[0][0];
    let mut min_y = boxes[0][1];
    let mut max_x = boxes[0][2];
    let mut max_y = boxes[0][3];

    for bbox in boxes.iter().skip(1) {
        min_x = min_x.min(bbox[0]);
        min_y = min_y.min(bbox[1]);
        max_x = max_x.max(bbox[2]);
        max_y = max_y.max(bbox[3]);
    }

    [min_x, min_y, max_x, max_y]
}

pub fn convert_to_corners(box_coords: &[i32; 4]) -> [i32; 4] {
    let [x_center, y_center, width, height] = *box_coords;
    [
        x_center - width / 2,  // x1
        y_center - height / 2, // y1
        x_center + width / 2,  // x2
        y_center + height / 2, // y2
    ]
}

pub fn calculate_iou(box1: &[i32; 4], box2: &[i32; 4]) -> f32 {
    let box1_corners = convert_to_corners(box1);
    let box2_corners = convert_to_corners(box2);

    let x_left = box1_corners[0].max(box2_corners[0]);
    let y_top = box1_corners[1].max(box2_corners[1]);
    let x_right = box1_corners[2].min(box2_corners[2]);
    let y_bottom = box1_corners[3].min(box2_corners[3]);

    if x_right <= x_left || y_bottom <= y_top {
        return 0.0;
    }

    let intersection_area = ((x_right - x_left) * (y_bottom - y_top)) as f32;
    let box1_area = (box1[2] * box1[3]) as f32;
    let box2_area = (box2[2] * box2[3]) as f32;
    let union_area = box1_area + box2_area - intersection_area;

    intersection_area / union_area
}
