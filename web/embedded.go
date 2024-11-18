package main

import (
	"embed"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

//go:embed lib/libonnxruntime.so.1.20.0
//go:embed onnx_model/yolo11n_9ir_256_haface.onnx
var embeddedFiles embed.FS

// extractFiles extracts library and model
func extractFiles(_ string) (string, string, error) {
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

    // Extract model
    modelPath, err := extractModel(tmpDir)
    if err != nil {
        os.RemoveAll(tmpDir)
        return "", "", err
    }

    return libPath, modelPath, nil
}

// extractLibrary extracts the ONNX Runtime library
func extractLibrary(tmpDir string) (string, error) {
    libName := "libonnxruntime.so.1.20.0"
    if runtime.GOOS == "darwin" {
        libName = "libonnxruntime.1.20.0.dylib"
    } else if runtime.GOOS == "windows" {
        libName = "onnxruntime.dll"
    }

    libData, err := embeddedFiles.Open(filepath.Join("lib", libName))
    if err != nil {
        return "", err
    }
    defer libData.Close()

    tmpLib := filepath.Join(tmpDir, libName)
    if err := extractFile(libData, tmpLib); err != nil {
        return "", err
    }

    if runtime.GOOS != "windows" {
        if err := os.Chmod(tmpLib, 0755); err != nil {
            return "", err
        }
    }

    return tmpLib, nil
}

// extractModel extracts the ONNX model
func extractModel(tmpDir string) (string, error) {
    modelName := "yolo11n_9ir_256_haface.onnx"

    modelData, err := embeddedFiles.Open(filepath.Join("onnx_model", modelName))
    if err != nil {
        return "", err
    }
    defer modelData.Close()

    tmpModel := filepath.Join(tmpDir, modelName)
    if err := extractFile(modelData, tmpModel); err != nil {
        return "", err
    }

    return tmpModel, nil
}

func extractFile(src io.Reader, destPath string) error {
    outFile, err := os.Create(destPath)
    if err != nil {
        return err
    }
    defer outFile.Close()

    _, err = io.Copy(outFile, src)
    return err
}
