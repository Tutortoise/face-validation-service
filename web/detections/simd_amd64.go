package detections

import (
	"unsafe"
)

//go:noescape
func processRowAVX512(dst, src unsafe.Pointer, width int)

//go:noescape
func processRowAVX2(dst, src unsafe.Pointer, width int)

//go:noescape
func processRowSSE41(dst, src unsafe.Pointer, width int)
