# =============================================================================
# FEA Compute Server - Multi-Stage Docker Build
# =============================================================================
#
# This Dockerfile creates a production-ready container for the FEA server.
# It uses a multi-stage build to minimize the final image size.
#
# Build:
#   docker build -t fea-server:latest .
#
# Run:
#   docker run -p 8080:8080 -v ./data:/data fea-server:latest
#
# =============================================================================

# =============================================================================
# Stage 1: Build deal.II and dependencies
# =============================================================================
FROM ubuntu:22.04 AS dealii-builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    pkg-config \
    # Fortran for LAPACK
    gfortran \
    # Boost libraries
    libboost-all-dev \
    # Linear algebra
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    # Sparse solvers
    libsuitesparse-dev \
    libmetis-dev \
    # HDF5 for I/O
    libhdf5-dev \
    # Math parsing
    libmuparser-dev \
    # Compression
    zlib1g-dev \
    # Parallel mesh
    libp4est-dev \
    # PETSc for parallel solvers
    petsc-dev \
    # MPI
    libopenmpi-dev \
    openmpi-bin \
    # SSL for downloads
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone and build deal.II
WORKDIR /opt

# Clone deal.II (specific version for reproducibility)
RUN git clone --depth 1 --branch v9.5.2 https://github.com/dealii/dealii.git dealii-src

# Build deal.II with required features
WORKDIR /opt/dealii-src/build
RUN cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/dealii \
    # Core features
    -DDEAL_II_WITH_LAPACK=ON \
    -DDEAL_II_WITH_UMFPACK=ON \
    -DDEAL_II_WITH_METIS=ON \
    # Parallel features
    -DDEAL_II_WITH_MPI=ON \
    -DDEAL_II_WITH_PETSC=ON \
    -DDEAL_II_WITH_P4EST=ON \
    # I/O features
    -DDEAL_II_WITH_HDF5=ON \
    -DDEAL_II_WITH_ZLIB=ON \
    # Utilities
    -DDEAL_II_WITH_MUPARSER=ON \
    -DDEAL_II_WITH_THREADS=ON \
    # Disable unnecessary components
    -DDEAL_II_COMPONENT_EXAMPLES=OFF \
    -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
    && ninja -j$(nproc) \
    && ninja install

# Clean up source to save space
RUN rm -rf /opt/dealii-src

# =============================================================================
# Stage 2: Build FEA Server
# =============================================================================
FROM dealii-builder AS fea-builder

# Copy source code
WORKDIR /app
COPY . .

# Build FEA server
RUN mkdir -p build && cd build && \
    cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DDEAL_II_DIR=/opt/dealii \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    && ninja -j$(nproc)

# Strip binary to reduce size
RUN strip build/fea_server

# =============================================================================
# Stage 3: Runtime Image (Minimal)
# =============================================================================
FROM ubuntu:22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Labels
LABEL maintainer="FEA Team"
LABEL version="1.0.0"
LABEL description="3D Static Structural FEA Compute Server"

# Install runtime dependencies only (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Boost runtime
    libboost-system1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-iostreams1.74.0 \
    libboost-serialization1.74.0 \
    # Linear algebra runtime
    liblapack3 \
    libblas3 \
    libopenblas0 \
    # Sparse solver runtime
    libumfpack5 \
    libmetis5 \
    libcholmod3 \
    libamd2 \
    libcamd2 \
    libcolamd2 \
    libccolamd2 \
    libsuitesparseconfig5 \
    # HDF5 runtime
    libhdf5-103-1 \
    # p4est runtime
    libp4est-2.2 \
    # PETSc runtime
    libpetsc-real3.15 \
    # MPI runtime
    libopenmpi3 \
    # Other runtime deps
    libmuparser2v5 \
    zlib1g \
    # Utilities
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy deal.II libraries
COPY --from=fea-builder /opt/dealii/lib /opt/dealii/lib

# Copy FEA server binary
COPY --from=fea-builder /app/build/fea_server /usr/local/bin/

# Create data directories
RUN mkdir -p /data/jobs /data/meshes /data/logs /data/temp

# Set library path
ENV LD_LIBRARY_PATH=/opt/dealii/lib:$LD_LIBRARY_PATH

# Expose HTTP port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget -q --spider http://localhost:8080/api/health || exit 1

# Create non-root user for security
RUN useradd -m -s /bin/bash -u 1000 fea && \
    chown -R fea:fea /data

# Switch to non-root user
USER fea

# Working directory
WORKDIR /data

# Default command
ENTRYPOINT ["fea_server"]
CMD ["--port", "8080", "--workers", "4", "--data-dir", "/data"]
