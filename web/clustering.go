package main

import (
	"math"
	"sort"
)

const (
	DefaultClusterSize = 50.0
	IouThreshold       = 0.45
)

func clusterBoxes(detections []Detection) [][4]int32 {
	if len(detections) == 0 {
		return nil
	}

	medianSize := calculateMedianSize(detections)
	eps := math.Max(medianSize, DefaultClusterSize) * 0.5
	minPoints := 1
	if len(detections) > 3 {
		minPoints = 2
	}

	points := make([][]float64, len(detections))
	for i, det := range detections {
		points[i] = []float64{
			float64(det.BBox[0]),
			float64(det.BBox[1]),
			float64(det.BBox[2]),
			float64(det.BBox[3]),
		}
	}

	clusters := dbscan(points, eps, minPoints)
	return processClusters(detections, clusters)
}

func calculateMedianSize(detections []Detection) float64 {
	sizes := make([]float64, len(detections))
	for i, det := range detections {
		width := float64(det.BBox[2] - det.BBox[0])
		height := float64(det.BBox[3] - det.BBox[1])
		sizes[i] = math.Sqrt(width * height)
	}

	sort.Float64s(sizes)
	if len(sizes) == 0 {
		return DefaultClusterSize
	}
	return sizes[len(sizes)/2]
}

func processClusters(detections []Detection, clusters []int) [][4]int32 {
	clusterMap := make(map[int][][4]int32)
	var finalBoxes [][4]int32

	// Group boxes by cluster
	for i, cluster := range clusters {
		if cluster == -1 {
			// Handle noise points
			bbox := detections[i].BBox
			merged := false
			for _, boxes := range clusterMap {
				for _, existing := range boxes {
					if calculateIOU(bbox, existing) > IouThreshold {
						boxes = append(boxes, bbox)
						merged = true
						break
					}
				}
				if merged {
					break
				}
			}
			if !merged {
				finalBoxes = append(finalBoxes, bbox)
			}
		} else {
			clusterMap[cluster] = append(clusterMap[cluster], detections[i].BBox)
		}
	}

	// Merge boxes in each cluster
	for _, boxes := range clusterMap {
		if len(boxes) > 0 {
			finalBoxes = append(finalBoxes, mergeBoxes(boxes))
		}
	}

	return finalBoxes
}

func calculateIOU(box1, box2 [4]int32) float64 {
	x1 := math.Max(float64(box1[0]), float64(box2[0]))
	y1 := math.Max(float64(box1[1]), float64(box2[1]))
	x2 := math.Min(float64(box1[2]), float64(box2[2]))
	y2 := math.Min(float64(box1[3]), float64(box2[3]))

	if x2 <= x1 || y2 <= y1 {
		return 0.0
	}

	intersection := (x2 - x1) * (y2 - y1)
	area1 := float64((box1[2] - box1[0]) * (box1[3] - box1[1]))
	area2 := float64((box2[2] - box2[0]) * (box2[3] - box2[1]))
	union := area1 + area2 - intersection

	return intersection / union
}

func mergeBoxes(boxes [][4]int32) [4]int32 {
	if len(boxes) == 0 {
		return [4]int32{0, 0, 0, 0}
	}

	result := boxes[0]
	for _, box := range boxes[1:] {
		result[0] = min32(result[0], box[0])
		result[1] = min32(result[1], box[1])
		result[2] = max32(result[2], box[2])
		result[3] = max32(result[3], box[3])
	}

	return result
}

func min32(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

func max32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

func dbscan(points [][]float64, eps float64, minPoints int) []int {
	n := len(points)
	clusters := make([]int, n)
	for i := range clusters {
		clusters[i] = -1 // Initialize all points as noise
	}

	currentCluster := 0
	for i := 0; i < n; i++ {
		if clusters[i] != -1 {
			continue
		}

		neighbors := getNeighbors(points, i, eps)
		if len(neighbors) < minPoints {
			continue
		}

		clusters[i] = currentCluster
		expandCluster(points, clusters, neighbors, currentCluster, eps, minPoints)
		currentCluster++
	}

	return clusters
}

func getNeighbors(points [][]float64, pointIdx int, eps float64) []int {
	var neighbors []int
	for i := range points {
		if distance(points[pointIdx], points[i]) <= eps {
			neighbors = append(neighbors, i)
		}
	}
	return neighbors
}

func expandCluster(points [][]float64, clusters []int, neighbors []int, cluster int, eps float64, minPoints int) {
	for i := 0; i < len(neighbors); i++ {
		pointIdx := neighbors[i]
		if clusters[pointIdx] == -1 {
			clusters[pointIdx] = cluster
			newNeighbors := getNeighbors(points, pointIdx, eps)
			if len(newNeighbors) >= minPoints {
				neighbors = append(neighbors, newNeighbors...)
			}
		}
	}
}

func distance(p1, p2 []float64) float64 {
	sum := 0.0
	for i := range p1 {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}
