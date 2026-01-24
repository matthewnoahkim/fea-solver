/**
 * Shared TypeScript types for FEA Backend
 */

// Unit Systems
export type UnitSystem = 'SI' | 'SI_MM' | 'US_CUSTOMARY';

// Material Types
export type MaterialType = 
  | 'linear_elastic'
  | 'orthotropic'
  | 'elastoplastic'
  | 'hyperelastic';

export interface LinearElasticMaterial {
  type: 'linear_elastic';
  E: number;          // Young's modulus
  nu: number;         // Poisson's ratio
  density?: number;
  yield_stress?: number;
  ultimate_strength?: number;
  thermal_expansion?: number;
}

export interface OrthotropicMaterial {
  type: 'orthotropic';
  E1: number;
  E2: number;
  E3: number;
  nu12: number;
  nu13: number;
  nu23: number;
  G12: number;
  G13: number;
  G23: number;
  density?: number;
  orientation?: {
    phi: number;
    theta: number;
    psi: number;
  };
}

export interface HyperelasticMaterial {
  type: 'hyperelastic';
  model: 'neo_hookean' | 'mooney_rivlin' | 'ogden';
  mu?: number;
  kappa?: number;
  C1?: number;
  C2?: number;
  density?: number;
}

export type Material = 
  | LinearElasticMaterial 
  | OrthotropicMaterial 
  | HyperelasticMaterial;

// Boundary Conditions
export type BCType = 
  | 'fixed'
  | 'displacement'
  | 'symmetry'
  | 'elastic_support'
  | 'roller';

export interface FixedBC {
  type: 'fixed';
  boundary_id: number;
}

export interface DisplacementBC {
  type: 'displacement';
  boundary_id: number;
  displacement: [number, number, number];
  components?: [boolean, boolean, boolean];
}

export interface SymmetryBC {
  type: 'symmetry';
  boundary_id: number;
  normal: [number, number, number];
}

export type BoundaryCondition = FixedBC | DisplacementBC | SymmetryBC;

// Loads
export type LoadType = 
  | 'pressure'
  | 'surface_force'
  | 'point_force'
  | 'gravity'
  | 'centrifugal'
  | 'thermal';

export interface PressureLoad {
  type: 'pressure';
  boundary_id: number;
  value: number;
}

export interface SurfaceForceLoad {
  type: 'surface_force';
  boundary_id: number;
  traction: [number, number, number];
}

export interface PointForceLoad {
  type: 'point_force';
  location: [number, number, number];
  force: [number, number, number];
}

export interface GravityLoad {
  type: 'gravity';
  acceleration: [number, number, number];
}

export type Load = 
  | PressureLoad 
  | SurfaceForceLoad 
  | PointForceLoad 
  | GravityLoad;

// Mesh
export interface MeshInput {
  format: 'gmsh' | 'vtk' | 'exodus' | 'abaqus';
  data?: string;  // Base64 encoded
  file_url?: string;
}

// Analysis Request
export interface AnalysisRequest {
  mesh: MeshInput;
  materials: Array<{
    id: number;
    properties: Material;
  }>;
  boundary_conditions: BoundaryCondition[];
  loads: Load[];
  settings?: AnalysisSettings;
}

export interface AnalysisSettings {
  unit_system?: UnitSystem;
  solver?: 'cg' | 'gmres' | 'direct';
  tolerance?: number;
  max_iterations?: number;
  nonlinear?: boolean;
  large_deformation?: boolean;
  output_vtk?: boolean;
}

// Analysis Results
export interface AnalysisResults {
  converged: boolean;
  iterations: number;
  final_residual: number;
  max_displacement: number;
  max_von_mises: number;
  min_safety_factor: number;
  total_strain_energy: number;
  timing: {
    total: number;
    assembly: number;
    solve: number;
    postprocess: number;
  };
  vtk_file?: string;
  warnings?: string[];
  error?: string;
}

// Job
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface Job {
  job_id: string;
  status: JobStatus;
  progress?: number;
  created_at: string;
  completed_at?: string;
  results?: AnalysisResults;
  error?: string;
}

// Mesh Quality
export interface MeshQualitySummary {
  total_elements: number;
  invalid_elements: number;
  poor_quality_elements: number;
  min_jacobian_ratio: number;
  max_aspect_ratio: number;
  avg_aspect_ratio: number;
  max_skewness: number;
  total_volume: number;
  warnings: string[];
  quality: 'good' | 'acceptable' | 'poor' | 'invalid';
}
