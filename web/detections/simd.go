package detections

import (
	"image"
	"runtime"
	"sync"
	"unsafe"

	"golang.org/x/sys/cpu"
)

var (
	useAVX512 = cpu.X86.HasAVX512
	useAVX2   = cpu.X86.HasAVX2
	useSSE41  = cpu.X86.HasSSE41
)

// SIMDPreprocessor handles optimized image preprocessing
type SIMDPreprocessor struct {
	width, height int
	channels      int
	numWorkers    int
	bufferPool    *sync.Pool
}

const CacheLineSize = 64

type alignedBuffer struct {
	data []float32
	pad  [CacheLineSize]byte
}

func NewSIMDPreprocessor() *SIMDPreprocessor {
	return &SIMDPreprocessor{
		width:      InputWidth,
		height:     InputHeight,
		channels:   3,
		numWorkers: runtime.GOMAXPROCS(0),
		bufferPool: &sync.Pool{
			New: func() interface{} {
				return &alignedBuffer{
					data: make([]float32, InputWidth*InputHeight*3),
				}
			},
		},
	}
}

func (p *SIMDPreprocessor) Process(img image.Image) []float32 {
	buffer := p.getAlignedBuffer()
	defer p.bufferPool.Put(buffer)

	switch {
	case useAVX512 && runtime.GOARCH == "amd64":
		p.processBufferWithSIMD(buffer.data, img, processRowAVX512)
	case useAVX2 && runtime.GOARCH == "amd64":
		p.processBufferWithSIMD(buffer.data, img, processRowAVX2)
	case useSSE41 && runtime.GOARCH == "amd64":
		p.processBufferWithSIMD(buffer.data, img, processRowSSE41)
	default:
		p.processGeneric(img, buffer.data)
	}

	return buffer.data
}

func (p *SIMDPreprocessor) getAlignedBuffer() *alignedBuffer {
	return p.bufferPool.Get().(*alignedBuffer)
}

func (p *SIMDPreprocessor) processBufferWithSIMD(buffer []float32, pic image.Image, processRow func(unsafe.Pointer, unsafe.Pointer, int)) {
	if img, ok := pic.(*image.RGBA); ok {
		channelSize := p.width * p.height
		for y := 0; y < p.height; y++ {
			srcRow := unsafe.Pointer(&img.Pix[y*img.Stride])
			dstR := unsafe.Pointer(&buffer[y*p.width])
			dstG := unsafe.Pointer(&buffer[channelSize+y*p.width])
			dstB := unsafe.Pointer(&buffer[channelSize*2+y*p.width])

			processRow(dstR, srcRow, p.width)
			processRow(dstG, unsafe.Pointer(uintptr(srcRow)+1), p.width)
			processRow(dstB, unsafe.Pointer(uintptr(srcRow)+2), p.width)
		}
	} else {
		p.processGeneric(pic, buffer)
	}
}

func (p *SIMDPreprocessor) processGeneric(img image.Image, buffer []float32) {
	channelSize := p.width * p.height
	for y := 0; y < p.height; y++ {
		for x := 0; x < p.width; x++ {
			i := y*p.width + x
			r, g, b, _ := img.At(x, y).RGBA()
			buffer[i] = float32(r>>8) / 255.0
			buffer[channelSize+i] = float32(g>>8) / 255.0
			buffer[channelSize*2+i] = float32(b>>8) / 255.0
		}
	}
}
