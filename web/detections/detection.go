package detections

import (
	"context"
	"errors"
	"fmt"
	"image"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/Tutortoise/face-validation-service/clustering"
	"github.com/Tutortoise/face-validation-service/models"

	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
)

type ModelSession struct {
	Session      *ort.AdvancedSession
	Input        *ort.Tensor[float32]
	Output       *ort.Tensor[float32]
	preprocessor *SIMDPreprocessor
}

func NewModelSession(session *ort.AdvancedSession, input, output *ort.Tensor[float32]) *ModelSession {
	return &ModelSession{
		Session:      session,
		Input:        input,
		Output:       output,
		preprocessor: NewSIMDPreprocessor(),
	}
}

func (m *ModelSession) Destroy() {
	if m.Session != nil {
		m.Session.Destroy()
	}
	if m.Input != nil {
		m.Input.Destroy()
	}
	if m.Output != nil {
		m.Output.Destroy()
	}
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
	bufferPool = sync.Pool{
		New: func() interface{} {
			return make([]float32, InputWidth*InputHeight*3)
		},
	}
)

func ProcessImage(ctx context.Context, img image.Image, model *ModelSession, timings *models.ProcessingTimings) ([][4]int32, error) {
	var lastErr error

	for attempt := 1; attempt <= RetryAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			boxes, err := processImageInternal(img, model, timings)
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

func processImageInternal(img image.Image, model *ModelSession, timings *models.ProcessingTimings) ([][4]int32, error) {
	resizeStart := time.Now()
	resized := imaging.Resize(img, InputWidth, InputHeight, imaging.Linear)
	timings.Resize = time.Since(resizeStart)

	// Prepare input buffer
	prepStart := time.Now()
	err := prepareInput(resized, model.Input)
	if err != nil {
		return nil, fmt.Errorf("prepare input buffer: %w", err)
	}
	timings.Preprocess = time.Since(prepStart)

	// Run inference
	inferStart := time.Now()
	err = model.Session.Run()
	if err != nil {
		return nil, fmt.Errorf("model inference: %w", err)
	}
	timings.Inference = time.Since(inferStart)

	// Process predictions
	postStart := time.Now()
	detections, err := processPredictions(model.Output.GetData(), img.Bounds().Dx(), img.Bounds().Dy())
	if err != nil {
		return nil, fmt.Errorf("process predictions: %w", err)
	}
	timings.Postprocess = time.Since(postStart)

	// Cluster boxes
	clusterStart := time.Now()
	boxes := clustering.ClusterBoxes(detections) // Make sure this function is exported
	timings.Clustering = time.Since(clusterStart)

	return boxes, nil
}

func processBufferGeneric(buffer []float32, pic image.Image, channelSize int) {
	for y := 0; y < InputHeight; y++ {
		offset := y * InputWidth
		for x := 0; x < InputWidth; x++ {
			i := offset + x
			r, g, b, _ := pic.At(x, y).RGBA()
			buffer[i] = float32(r>>8) / 255.0
			buffer[channelSize+i] = float32(g>>8) / 255.0
			buffer[channelSize*2+i] = float32(b>>8) / 255.0
		}
	}
}

var preprocessor = NewSIMDPreprocessor()

func prepareInput(pic image.Image, dst *ort.Tensor[float32]) error {
	preprocessor := NewSIMDPreprocessor()
	buffer := preprocessor.Process(pic)
	copy(dst.GetData(), buffer)
	return nil
}

func (p *SIMDPreprocessor) processParallel(img image.Image, buffer []float32) {
	channelSize := p.width * p.height
	rowsPerWorker := p.height / p.numWorkers

	var wg sync.WaitGroup
	wg.Add(p.numWorkers)

	for w := 0; w < p.numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := (w + 1) * rowsPerWorker
		if w == p.numWorkers-1 {
			endRow = p.height
		}

		go func(start, end int) {
			defer wg.Done()
			for y := start; y < end; y++ {
				offset := y * p.width
				for x := 0; x < p.width; x++ {
					i := offset + x
					r, g, b, _ := img.At(x, y).RGBA()
					buffer[i] = float32(r>>8) / 255.0
					buffer[channelSize+i] = float32(g>>8) / 255.0
					buffer[channelSize*2+i] = float32(b>>8) / 255.0
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
}

func processPredictions(predictions []float32, originalWidth, originalHeight int) ([]models.Detection, error) {
	numPredictions := 1344
	threshold := float32(ConfThreshold)

	// Verify prediction array length
	expectedSize := 6 * 1344 // 6 channels * 1344 predictions
	if len(predictions) != expectedSize {
		return nil, fmt.Errorf("unexpected predictions length: got %d, want %d", len(predictions), expectedSize)
	}

	detections := make([]models.Detection, 0, 100)
	const chunkSize = 512
	numWorkers := runtime.NumCPU()
	jobs := make(chan int, numWorkers)
	results := make(chan []models.Detection, numWorkers)

	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localDetections := make([]models.Detection, 0, 100)

			for start := range jobs {
				end := start + chunkSize
				if end > numPredictions {
					end = numPredictions
				}

				for i := start; i < end; i++ {
					// Get confidence score from the 5th channel
					confidence := predictions[4*1344+i]
					if confidence >= threshold {
						bbox := calculateBBox(
							[]float32{
								predictions[i],        // x
								predictions[1344+i],   // y
								predictions[2*1344+i], // w
								predictions[3*1344+i], // h
							},
							float32(originalWidth),
							float32(originalHeight),
						)
						localDetections = append(localDetections, models.Detection{
							BBox:       bbox,
							Confidence: confidence,
						})
					}
				}
			}

			if len(localDetections) > 0 {
				results <- localDetections
			}
		}()
	}

	// Send jobs
	go func() {
		for i := 0; i < numPredictions; i += chunkSize {
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
	for detectionChunk := range results {
		detections = append(detections, detectionChunk...)
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
		int32(max32(0, x1)),
		int32(max32(0, y1)),
		int32(minF32(origWidth, x2)),
		int32(minF32(origHeight, y2)),
	}
}

func sortDetectionsByConfidence(detections []models.Detection) {
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Confidence > detections[j].Confidence
	})
}

func minF32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
