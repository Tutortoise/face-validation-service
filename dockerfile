# syntax=docker/dockerfile:1.4
# Build stage
FROM golang:1.22-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    upx \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only go.mod and go.sum first
COPY web/go.mod web/go.sum ./

# Download dependencies
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# Copy necessary source files
COPY web/ ./

# Build with optimizations
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=1 GOOS=linux \
    go build -trimpath \
    -ldflags='-s -w' \
    -o server && \
    upx --best --lzma server

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create directories and nonroot user
RUN mkdir -p /usr/lib && \
    groupadd -r nonroot && \
    useradd -r -g nonroot nonroot

# Copy files from builder
COPY --from=builder --chown=nonroot:nonroot /build/lib/libonnxruntime.so.1.20.0 /usr/lib/
COPY --from=builder --chown=nonroot:nonroot /build/server /server

# Set environment variables
ENV PORT=8080 \
    LD_LIBRARY_PATH=/usr/lib \
    GOMAXPROCS=2 \
    GOMEMLIMIT=1024MiB \
    TZ=UTC

# Expose port
EXPOSE 8080

# Configure health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/server", "-health-check"] || exit 1

# Set up dynamic linker cache
RUN ldconfig

# Use nonroot user
USER nonroot:nonroot

# Run the application
ENTRYPOINT ["/server"]
