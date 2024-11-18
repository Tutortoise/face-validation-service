package main

import (
	"context"
	"fmt"
	"github.com/Tutortoise/face-validation-service/detections"
	"sync"
	"time"
)

const (
	// DefaultPoolSize Pool configuration
	DefaultPoolSize   = 4
	AcquireTimeout    = 5 * time.Second
	HealthCheckPeriod = 60 * time.Second
)

type ModelSessionPool struct {
	sessions   chan *detections.ModelSession
	size       int
	modelPath  string
	mu         sync.Mutex
	closed     bool
	metrics    *PoolMetrics
	lastErrors []error
}

type PoolMetrics struct {
	mu              sync.RWMutex
	inUse           int
	totalAcquired   int64
	totalReleased   int64
	acquireFailures int64
	waitTime        time.Duration
}

func NewModelSessionPool(modelPath string, size int) (*ModelSessionPool, error) {
	if size <= 0 {
		size = DefaultPoolSize
	}

	pool := &ModelSessionPool{
		sessions:  make(chan *detections.ModelSession, size),
		size:      size,
		modelPath: modelPath,
		metrics:   &PoolMetrics{},
	}

	// Initialize sessions
	for i := 0; i < size; i++ {
		session, err := initSession(modelPath)
		if err != nil {
			pool.Destroy()
			return nil, fmt.Errorf("failed to initialize session %d: %w", i, err)
		}
		pool.sessions <- session
	}

	// Start health check routine
	go pool.healthCheck()

	return pool, nil
}

func (p *ModelSessionPool) Acquire(ctx context.Context) (*detections.ModelSession, error) {
	if p.closed {
		return nil, fmt.Errorf("pool is closed")
	}

	start := time.Now()
	defer func() {
		p.metrics.mu.Lock()
		p.metrics.waitTime += time.Since(start)
		p.metrics.mu.Unlock()
	}()

	select {
	case session := <-p.sessions:
		p.metrics.mu.Lock()
		p.metrics.inUse++
		p.metrics.totalAcquired++
		p.metrics.mu.Unlock()
		return session, nil
	case <-time.After(AcquireTimeout):
		p.metrics.mu.Lock()
		p.metrics.acquireFailures++
		p.metrics.mu.Unlock()
		return nil, fmt.Errorf("timeout waiting for available session")
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (p *ModelSessionPool) Release(session *detections.ModelSession) {
	if p.closed {
		session.Destroy()
		return
	}

	p.metrics.mu.Lock()
	p.metrics.inUse--
	p.metrics.totalReleased++
	p.metrics.mu.Unlock()

	p.sessions <- session
}

func (p *ModelSessionPool) Destroy() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return
	}

	p.closed = true
	close(p.sessions)

	// Destroy all sessions
	for session := range p.sessions {
		session.Destroy()
	}
}

func (p *ModelSessionPool) healthCheck() {
	ticker := time.NewTicker(HealthCheckPeriod)
	defer ticker.Stop()

	for range ticker.C {
		if p.closed {
			return
		}

		p.mu.Lock()
		currentSize := len(p.sessions)
		p.mu.Unlock()

		// Check if we need to recreate any sessions
		if currentSize < p.size {
			p.replenishSessions(p.size - currentSize)
		}
	}
}

func (p *ModelSessionPool) replenishSessions(count int) {
	for i := 0; i < count; i++ {
		session, err := initSession(p.modelPath)
		if err != nil {
			p.recordError(err)
			continue
		}
		p.sessions <- session
	}
}

func (p *ModelSessionPool) recordError(err error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.lastErrors = append(p.lastErrors, err)
	if len(p.lastErrors) > 10 {
		p.lastErrors = p.lastErrors[1:]
	}
}

func (p *ModelSessionPool) GetMetrics() PoolMetrics {
	p.metrics.mu.RLock()
	defer p.metrics.mu.RUnlock()
	return *p.metrics
}
