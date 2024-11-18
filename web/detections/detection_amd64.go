//go:build amd64
// +build amd64

package detections

import (
	"unsafe"
)

//go:noescape
func processRowAVX2(dst, src unsafe.Pointer, width int)
