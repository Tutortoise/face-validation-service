//go:build !amd64

package detections

import "unsafe"

func processRowAVX512(dst, src unsafe.Pointer, width int) {
    panic("AVX-512 not supported on this platform")
}

func processRowAVX2(dst, src unsafe.Pointer, width int) {
    panic("AVX2 not supported on this platform")
}

func processRowSSE41(dst, src unsafe.Pointer, width int) {
    panic("SSE4.1 not supported on this platform")
}
