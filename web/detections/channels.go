package detections

import (
	"image"
	"sync"
)

type channelProcessor struct {
	width, height int
	buffer        []float32
	channelSize   int
}

func newChannelProcessor(width, height int) *channelProcessor {
	return &channelProcessor{
		width:       width,
		height:      height,
		channelSize: width * height,
		buffer:      make([]float32, width*height*3),
	}
}

func (cp *channelProcessor) processChannels(img image.Image) {
	var wg sync.WaitGroup
	wg.Add(3)

	// Process each channel concurrently
	for c := 0; c < 3; c++ {
		go func(channel int) {
			defer wg.Done()
			offset := channel * cp.channelSize
			for y := 0; y < cp.height; y++ {
				for x := 0; x < cp.width; x++ {
					r, g, b, _ := img.At(x, y).RGBA()
					switch channel {
					case 0:
						cp.buffer[offset+y*cp.width+x] = float32(r>>8) / 255.0
					case 1:
						cp.buffer[offset+y*cp.width+x] = float32(g>>8) / 255.0
					case 2:
						cp.buffer[offset+y*cp.width+x] = float32(b>>8) / 255.0
					}
				}
			}
		}(c)
	}

	wg.Wait()
}
