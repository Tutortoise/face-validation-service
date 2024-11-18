package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
	"time"

	"github.com/Tutortoise/face-validation-service/detections"
	"github.com/Tutortoise/face-validation-service/models"

	"github.com/gorilla/mux"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	debugMode           bool
	modelOutputChannels = int64(6)
	modelOutputGridSize = int64(1344)
)

func init() {
	debugMode = os.Getenv("DEBUG") == "true"
	if envChannels := os.Getenv("MODEL_OUTPUT_CHANNELS"); envChannels != "" {
		if c, err := strconv.ParseInt(envChannels, 10, 64); err == nil {
			modelOutputChannels = c
		}
	}
	if envGridSize := os.Getenv("MODEL_OUTPUT_GRID_SIZE"); envGridSize != "" {
		if g, err := strconv.ParseInt(envGridSize, 10, 64); err == nil {
			modelOutputGridSize = g
		}
	}
}

func logTimings(t *models.ProcessingTimings) {
	if debugMode {
		log.Printf("[DEBUG] RequestID: %s - Processing times:\n"+
			"\tImage Decode: %v\n"+
			"\tResize:      %v\n"+
			"\tPreprocess:  %v\n"+
			"\tInference:   %v\n"+
			"\tPostprocess: %v\n"+
			"\tClustering:  %v\n"+
			"\tTotal:       %v",
			t.RequestID,
			t.ImageDecode,
			t.Resize,
			t.Preprocess,
			t.Inference,
			t.Postprocess,
			t.Clustering,
			t.Total)
	}
}

func initSession(modelPath string) (*detections.ModelSession, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("error creating session options: %w", err)
	}
	defer options.Destroy()

	options.SetIntraOpNumThreads(runtime.NumCPU())
	options.SetInterOpNumThreads(runtime.NumCPU())
	options.SetMemPattern(true)
	options.SetCpuMemArena(true)

	inputShape := ort.NewShape(1, 3, detections.InputWidth, detections.InputHeight)
	outputShape := ort.NewShape(1, modelOutputChannels, modelOutputGridSize) // Use configurable values

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("error creating input tensor: %w", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("error creating output tensor: %w", err)
	}

	// Create optimized session
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		options,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("error creating session: %w", err)
	}

	return &detections.ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}, nil
}

type AppState struct {
	ModelPath string
	Pool      *ModelSessionPool
}

type ValidationResponse struct {
	IsValid   bool   `json:"is_valid"`
	FaceCount int    `json:"face_count"`
	Message   string `json:"message"`
}

type ErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

func main() {
	// Add basic logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Set up model path
	modelPath := filepath.Clean("../models/yolo11n_9ir_256_haface.onnx")
	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		log.Fatalf("Failed to get absolute path for model: %v", err)
	}

	// Extract embedded files
	libPath, modelPath, err := extractFiles(absModelPath)
	if err != nil {
		log.Fatalf("Failed to extract embedded files: %v", err)
	}
	tmpDir := filepath.Dir(libPath)
	defer os.RemoveAll(tmpDir) // Clean up temporary directory

	// Initialize ONNX Runtime
	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Create session
	modelSession, err := initSession(modelPath)
	if err != nil {
		log.Fatalf("Failed to create model session: %v", err)
	}
	defer modelSession.Destroy()
	pool, err := NewModelSessionPool(modelPath, DefaultPoolSize)
	if err != nil {
		log.Fatalf("Failed to create model session pool: %v", err)
	}
	defer pool.Destroy()

	state := &AppState{
		ModelPath: modelPath,
		Pool:      pool,
	}

	r := mux.NewRouter()
	r.HandleFunc("/validate-face", handleValidateFace(state)).Methods("POST")
	state.addMonitoringRoutes(r)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	srv := &http.Server{
		Handler:      r,
		Addr:         "0.0.0.0:" + port,
		WriteTimeout: 60 * time.Second,
		ReadTimeout:  60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-stop
		log.Println("Shutting down gracefully...")

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := srv.Shutdown(ctx); err != nil {
			log.Printf("Error during shutdown: %v", err)
		}

		pool.Destroy()
		ort.DestroyEnvironment()
	}()

	log.Printf("Starting server on %s", srv.Addr)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}

func handleValidateFace(state *AppState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		startTotal := time.Now()
		requestID := fmt.Sprintf("%d", time.Now().UnixNano())
		timings := &models.ProcessingTimings{RequestID: requestID}

		ctx := r.Context()
		contentType := r.Header.Get("Content-Type")

		var imgBytes []byte
		var err error

		switch {
		case contentType == "application/json":
			imgBytes, err = handleJSONRequest(r)
		case contentType == "multipart/form-data":
			imgBytes, err = handleMultipartRequest(r)
		default:
			imgBytes, err = handleRawRequest(r)
		}

		if err != nil {
			sendErrorResponse(w, "invalid_request", err.Error(), http.StatusBadRequest)
			return
		}

		// Decode image
		decodeStart := time.Now()
		img, err := decodeImage(imgBytes)
		timings.ImageDecode = time.Since(decodeStart)
		if err != nil {
			sendErrorResponse(w, "invalid_image", "Failed to decode image", http.StatusBadRequest)
			return
		}

		// Acquire session from pool
		session, err := state.Pool.Acquire(ctx)
		if err != nil {
			sendErrorResponse(w, "session_error", err.Error(), http.StatusServiceUnavailable)
			return
		}
		defer state.Pool.Release(session)

		// Process image using the acquired session
		boxes, err := detections.ProcessImage(ctx, img, session, timings)
		if err != nil {
			sendErrorResponse(w, "processing_error", err.Error(), http.StatusInternalServerError)
			return
		}

		timings.Total = time.Since(startTotal)
		logTimings(timings)

		// Create response
		faceCount := len(boxes)
		response := ValidationResponse{
			IsValid:   faceCount == 1,
			FaceCount: faceCount,
			Message:   getFaceValidationMessage(faceCount),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func (s *AppState) addMonitoringRoutes(r *mux.Router) {
	r.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	}).Methods("GET")
	r.HandleFunc("/metrics", s.handleMetrics).Methods("GET")
}

func (s *AppState) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	metrics := s.Pool.GetMetrics()
	response := map[string]interface{}{
		"pool_size":        s.Pool.size,
		"sessions_in_use":  metrics.inUse,
		"total_acquired":   metrics.totalAcquired,
		"total_released":   metrics.totalReleased,
		"acquire_failures": metrics.acquireFailures,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handleJSONRequest(r *http.Request) ([]byte, error) {
	var req struct {
		Image string `json:"image"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(req.Image)
}

func handleMultipartRequest(r *http.Request) ([]byte, error) {
	if err := r.ParseMultipartForm(10 << 20); err != nil {
		return nil, err
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return io.ReadAll(file)
}

func handleRawRequest(r *http.Request) ([]byte, error) {
	return io.ReadAll(r.Body)
}

func decodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	return img, err
}

func getFaceValidationMessage(faceCount int) string {
	switch {
	case faceCount == 0:
		return "No faces detected"
	case faceCount == 1:
		return "Valid single face detected"
	default:
		return fmt.Sprintf("Multiple faces detected: %d", faceCount)
	}
}

func sendErrorResponse(w http.ResponseWriter, code, message string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(ErrorResponse{
		Code:    code,
		Message: message,
	})
}
