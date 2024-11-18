//go:build !amd64
// +build !amd64

package detections

import (
	"image"
	"unsafe"
)

func processRowAVX2(dst, src unsafe.Pointer, width int) {
	// This should never be called on non-amd64 platforms
	// The processBufferAVX2 function should use processBufferGeneric instead
	panic("processRowAVX2 called on unsupported platform")
}

func processBufferAVX2(buffer []float32, pic image.Image, channelSize int) {
	// Fallback to generic implementation on non-amd64 platforms
	processBufferGeneric(buffer, pic, channelSize)
}
