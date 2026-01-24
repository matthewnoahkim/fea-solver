/**
 * FEA API Gateway - Root Endpoint
 * 
 * Provides API information and documentation links.
 */

export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  
  res.status(200).json({
    service: 'FEA Analysis API Gateway',
    version: '1.0.0',
    description: '3D Static Structural Finite Element Analysis',
    documentation: '/docs',
    endpoints: {
      analyze: {
        method: 'POST',
        path: '/api/analyze',
        description: 'Submit a new analysis job'
      },
      job_status: {
        method: 'GET',
        path: '/api/jobs/{id}',
        description: 'Get job status'
      },
      job_results: {
        method: 'GET',
        path: '/api/jobs/{id}/results',
        description: 'Get analysis results'
      },
      job_files: {
        method: 'GET',
        path: '/api/jobs/{id}/files/{filename}',
        description: 'Download output files (VTK, CSV)'
      },
      cancel_job: {
        method: 'DELETE',
        path: '/api/jobs/{id}',
        description: 'Cancel a queued job'
      },
      mesh_quality: {
        method: 'POST',
        path: '/api/mesh/quality',
        description: 'Analyze mesh quality'
      },
      materials: {
        method: 'GET',
        path: '/api/materials',
        description: 'List available materials'
      },
      health: {
        method: 'GET',
        path: '/api/health',
        description: 'Server health check'
      }
    },
    support: {
      mesh_formats: ['GMSH (.msh)', 'VTK (.vtk, .vtu)', 'Abaqus (.inp)'],
      material_models: ['Isotropic Elastic', 'Orthotropic', 'Elastoplastic', 'Hyperelastic'],
      boundary_conditions: ['Fixed', 'Displacement', 'Symmetry', 'Elastic Support'],
      load_types: ['Surface Force', 'Pressure', 'Point Force', 'Gravity', 'Thermal', 'Centrifugal'],
      output_formats: ['JSON', 'VTK/VTU', 'CSV']
    }
  });
}
