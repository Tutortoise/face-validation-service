package models

import "time"

type Detection struct {
	BBox       [4]int32
	Confidence float32
}

type ProcessingTimings struct {
	RequestID   string
	ImageDecode time.Duration
	Resize      time.Duration
	Preprocess  time.Duration
	Inference   time.Duration
	Postprocess time.Duration
	Clustering  time.Duration
	Total       time.Duration
}
