package main

import (
	"context"
	"errors"
	"fmt"
	"image"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
)

type Detection struct {
	BBox       [4]int32
	Confidence float32
}

type ProcessingError struct {
	Message string
	Cause   error
}

func (e *ProcessingError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

var (
	inputBufferCache sync.Map
	ErrTimeout       = errors.New("processing timeout")
)

var bufferPool = sync.Pool{
	New: func() interface{} {
		return make([]float32, InputWidth*InputHeight*3)
	},
}

func ProcessImage(ctx context.Context, img image.Image, model *ModelSession) ([][4]int32, error) {
	var lastErr error

	for attempt := 1; attempt <= RetryAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			boxes, err := processImageInternal(img, model)
			if err == nil {
				return boxes, nil
			}
			lastErr = err

			if attempt < RetryAttempts {
				time.Sleep(time.Duration(attempt) * RetryDelayMs * time.Millisecond)
				continue
			}
		}
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("unknown error")
}

func processImageInternal(img image.Image, model *ModelSession) ([][4]int32, error) {
	// Resize image
	resized := imaging.Resize(img, InputWidth, InputHeight, imaging.Lanczos)

	// Prepare input buffer
	err := prepareInput(resized, model.Input)
	if err != nil {
		return nil, fmt.Errorf("prepare input buffer: %w", err)
	}

	// Run inference
	err = model.Session.Run()
	if err != nil {
		return nil, fmt.Errorf("model inference: %w", err)
	}

	// Process predictions
	detections, err := processPredictions(model.Output.GetData(), img.Bounds().Dx(), img.Bounds().Dy())
	if err != nil {
		return nil, fmt.Errorf("process predictions: %w", err)
	}

	return clusterBoxes(detections), nil
}

func prepareInput(pic image.Image, dst *ort.Tensor[float32]) error {
	data := dst.GetData()
	channelSize := InputWidth * InputHeight

	// Use sync.Pool for buffers
	buffer := bufferPool.Get().([]float32)
	defer bufferPool.Put(buffer)

	// Process image using multiple goroutines
	numWorkers := runtime.NumCPU()
	rowsPerWorker := InputHeight / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		startY := w * rowsPerWorker
		endY := startY + rowsPerWorker
		if w == numWorkers-1 {
			endY = InputHeight
		}

		go func(startY, endY int) {
			defer wg.Done()
			for y := startY; y < endY; y++ {
				offset := y * InputWidth
				for x := 0; x < InputWidth; x++ {
					i := offset + x
					r, g, b, _ := pic.At(x, y).RGBA()
					buffer[i] = float32(r>>8) / 255.0
					buffer[channelSize+i] = float32(g>>8) / 255.0
					buffer[channelSize*2+i] = float32(b>>8) / 255.0
				}
			}
		}(startY, endY)
	}

	wg.Wait()

	// Efficient copy to tensor
	copy(data, buffer)
	return nil
}

func processPredictions(predictions []float32, originalWidth, originalHeight int) ([]Detection, error) {
	// Pre-allocate slice with capacity
	detections := make([]Detection, 0, 100)
	numPredictions := 8400

	// Create worker pool
	numWorkers := runtime.NumCPU()
	jobs := make(chan int, numPredictions)
	results := make(chan Detection, numPredictions)
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				confidence := predictions[4*8400+i]
				if confidence >= ConfThreshold {
					bbox := calculateBBox(
						[]float32{
							predictions[i],
							predictions[8400+i],
							predictions[2*8400+i],
							predictions[3*8400+i],
						},
						float32(originalWidth),
						float32(originalHeight),
					)
					results <- Detection{
						BBox:       bbox,
						Confidence: confidence,
					}
				}
			}
		}()
	}

	// Send jobs
	go func() {
		for i := 0; i < numPredictions; i++ {
			jobs <- i
		}
		close(jobs)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	// Gather results
	for detection := range results {
		detections = append(detections, detection)
	}

	if len(detections) > 0 {
		sortDetectionsByConfidence(detections)
	}

	return detections, nil
}

func calculateBBox(coords []float32, origWidth, origHeight float32) [4]int32 {
	// Scale factors
	scaleX := origWidth / InputWidth
	scaleY := origHeight / InputHeight

	// Convert center coordinates to corners
	centerX := coords[0] * InputWidth
	centerY := coords[1] * InputHeight
	width := coords[2] * InputWidth
	height := coords[3] * InputHeight

	x1 := (centerX - width/2) * scaleX
	y1 := (centerY - height/2) * scaleY
	x2 := (centerX + width/2) * scaleX
	y2 := (centerY + height/2) * scaleY

	return [4]int32{
		int32(max(0, x1)),
		int32(max(0, y1)),
		int32(min(float32(origWidth), x2)),
		int32(min(float32(origHeight), y2)),
	}
}

func sortDetectionsByConfidence(detections []Detection) {
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Confidence > detections[j].Confidence
	})
}

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
