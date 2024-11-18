package main

import (
	"embed"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

//go:embed lib/libonnxruntime.so.1.20.0
var embeddedFiles embed.FS

// extractFiles extracts library and sets up model path
func extractFiles(modelPath string) (string, string, error) {
	// Validate model path
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return "", "", fmt.Errorf("model file not found: %s", modelPath)
	}

	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "face-validation")
	if err != nil {
		return "", "", err
	}

	// Extract library
	libPath, err := extractLibrary(tmpDir)
	if err != nil {
		os.RemoveAll(tmpDir)
		return "", "", err
	}

	return libPath, modelPath, nil
}

// extractLibrary extracts the ONNX Runtime library
func extractLibrary(tmpDir string) (string, error) {
	// Determine library name based on OS
	libName := "libonnxruntime.so.1.20.0"
	if runtime.GOOS == "darwin" {
		libName = "libonnxruntime.1.20.0.dylib"
	} else if runtime.GOOS == "windows" {
		libName = "onnxruntime.dll"
	}

	// Read embedded library
	libData, err := embeddedFiles.Open(filepath.Join("lib", libName))
	if err != nil {
		return "", err
	}
	defer libData.Close()

	// Create temporary file for the library
	tmpLib := filepath.Join(tmpDir, libName)
	if err := extractFile(libData, tmpLib); err != nil {
		return "", err
	}

	// Make the library executable
	if runtime.GOOS != "windows" {
		if err := os.Chmod(tmpLib, 0755); err != nil {
			return "", err
		}
	}

	return tmpLib, nil
}

// extractFile is a helper function to extract a file
func extractFile(src io.Reader, destPath string) error {
	outFile, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, src)
	return err
}
