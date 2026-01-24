# FEA Backend - 3D Static Structural Finite Element Analysis

A production-grade 3D static structural finite element analysis backend using [deal.II](https://www.dealii.org/). This system provides a REST API for submitting and managing FEA analysis jobs.

## Architecture

```
┌──────────────────┐     ┌────────────────────────────────────────┐
│                  │     │           Compute Server               │
│   Vercel API     │────▶│  ┌─────────────────────────────────┐   │
│   Gateway        │     │  │        HTTP Server              │   │
│  (Node.js)       │     │  │     (cpp-httplib)               │   │
│                  │     │  └─────────────────────────────────┘   │
└──────────────────┘     │              │                         │
                         │              ▼                         │
                         │  ┌─────────────────────────────────┐   │
                         │  │      Job Queue + Workers        │   │
                         │  │   (Thread Pool)                 │   │
                         │  └─────────────────────────────────┘   │
                         │              │                         │
                         │              ▼                         │
                         │  ┌─────────────────────────────────┐   │
                         │  │      ElasticProblem<3>          │   │
                         │  │   (deal.II FE Solver)           │   │
                         │  └─────────────────────────────────┘   │
                         └────────────────────────────────────────┘
```

## Features

### Material Models
- **Isotropic Linear Elastic** - Standard metals, polymers
- **Orthotropic Elastic** - Composites, wood
- **Elastoplastic (von Mises)** - Metals beyond yield
- **Hyperelastic** - Neo-Hookean, Mooney-Rivlin for rubbers

### Boundary Conditions
- Fixed constraints
- Prescribed displacements
- Symmetry planes
- Elastic supports
- Frictionless contact

### Loads
- Surface forces & pressure
- Point forces & moments
- Remote forces
- Gravity & linear acceleration
- Centrifugal loads
- Thermal loads

### Connections
- Springs (to ground, two-point, bushing)
- Rigid links (RBE2, RBE3)
- Tied contact
- Multi-point constraints

### Post-Processing
- Von Mises & principal stresses
- Reaction forces & equilibrium check
- Safety factors (multiple criteria)
- Linearized stress (ASME BPVC)
- Strain energy
- VTK output for visualization

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OR: deal.II 9.4+, CMake 3.13+, C++17 compiler

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourorg/fea-backend.git
cd fea-backend/compute-server

# Build and run
docker-compose up -d

# Check health
curl http://localhost:8080/api/health
```

### Building from Source

```bash
cd compute-server

# Create build directory
mkdir build && cd build

# Configure (adjust DEAL_II_DIR as needed)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDEAL_II_DIR=/path/to/dealii

# Build
make -j$(nproc)

# Run
./fea_server --port 8080 --workers 4 --data-dir ./data
```

## API Reference

### Submit Analysis Job

```bash
POST /api/analyze
Content-Type: application/json

{
  "mesh": {
    "type": "box",
    "min": [0, 0, 0],
    "max": [100, 10, 10],
    "subdivisions": [20, 4, 4]
  },
  "materials": {
    "default": "steel_structural"
  },
  "boundary_conditions": [
    {
      "type": "fixed",
      "target": {"type": "boundary_id", "id": 0}
    }
  ],
  "loads": [
    {
      "type": "surface_force",
      "target": {"type": "boundary_id", "id": 1},
      "force_per_area": [0, 0, -1000000]
    }
  ],
  "solver_options": {
    "fe_degree": 2,
    "refinement_cycles": 2
  }
}
```

Response:
```json
{
  "job_id": "fea_20260124_123456_789012",
  "status": "queued",
  "queue_position": 1
}
```

### Check Job Status

```bash
GET /api/jobs/{job_id}
```

Response:
```json
{
  "job_id": "fea_20260124_123456_789012",
  "status": "running",
  "progress": 0.45,
  "current_stage": "Assembling system"
}
```

### Get Results

```bash
GET /api/jobs/{job_id}/results
```

Response:
```json
{
  "job_id": "fea_20260124_123456_789012",
  "status": "completed",
  "displacements": {
    "max": {"x": 0.001, "y": 0.002, "z": -0.015},
    "min": {"x": -0.001, "y": -0.002, "z": 0.0}
  },
  "stress": {
    "von_mises": {"max": 125000000, "avg": 45000000},
    "principal": {"max": 130000000, "min": -25000000}
  },
  "reactions": {
    "total_force": [0, 0, 10000],
    "total_moment": [500000, 0, 0]
  },
  "safety_factors": {
    "min": 2.0,
    "avg": 5.5
  }
}
```

### Download Output Files

```bash
GET /api/jobs/{job_id}/files/results.vtu
```

### List Materials

```bash
GET /api/materials
```

### Health Check

```bash
GET /api/health
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FEA_PORT` | Server port | 8080 |
| `FEA_WORKERS` | Worker threads | 4 |
| `FEA_DATA_DIR` | Data directory | /data |
| `FEA_LOG_LEVEL` | Log level | info |

### Command Line Options

```
fea_server [options]
  --port <port>       Server port (default: 8080)
  --workers <n>       Worker threads (default: 4)
  --data-dir <path>   Data directory (default: /data)
  --help              Show help
```

## Unit Systems

Supported unit systems:
- **SI** - meters, Newtons, Pascals, Kelvin
- **SI_MM** - millimeters, Newtons, MPa, Celsius
- **US_CUSTOMARY** - inches, pounds-force, psi, Fahrenheit

All internal calculations use SI units.

```json
{
  "units": {
    "type": "SI_MM"
  }
}
```

## Deployment

### Vercel API Gateway

```bash
# Install Vercel CLI
npm i -g vercel

# Set compute server URL
vercel secrets add compute_server_url "http://your-server:8080"

# Deploy
cd fea-backend
vercel --prod
```

### Cloud VM (AWS/GCP/DigitalOcean)

```bash
# Pull and run Docker image
docker run -d \
  --name fea-server \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /data/fea:/data \
  -e FEA_WORKERS=8 \
  fea-server:latest
```

## Project Structure

```
fea-backend/
├── compute-server/
│   ├── src/
│   │   ├── main.cc              # Entry point
│   │   ├── api/
│   │   │   ├── http_server.h    # HTTP server
│   │   │   ├── json_parser.h    # JSON utilities
│   │   │   └── request_validator.h
│   │   ├── solver/
│   │   │   ├── elastic_problem.h # Main solver
│   │   │   ├── material_library.h
│   │   │   ├── boundary_conditions.h
│   │   │   ├── loads/
│   │   │   ├── connections/
│   │   │   └── nonlinear/
│   │   ├── mesh/
│   │   │   ├── mesh_reader.h
│   │   │   └── mesh_quality.h
│   │   └── post/
│   │       ├── stress_calculator.h
│   │       ├── reaction_forces.h
│   │       ├── safety_factors.h
│   │       └── linearized_stress.h
│   ├── CMakeLists.txt
│   ├── Dockerfile
│   └── docker-compose.yml
├── api/                          # Vercel serverless functions
│   ├── analyze.js
│   ├── jobs/
│   └── health.js
├── vercel.json
└── README.md
```

## Testing

### Cantilever Beam Validation

Compare tip deflection against analytical solution:
```
δ = PL³/(3EI)
```

```bash
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d @tests/validation/cantilever_beam.json
```

### Run Unit Tests

```bash
cd compute-server/build
cmake .. -DBUILD_TESTS=ON
make
ctest --output-on-failure
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [deal.II](https://www.dealii.org/) - Finite element library
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP server
- [nlohmann/json](https://github.com/nlohmann/json) - JSON library
