/**
 * Validation schemas for FEA API requests
 */

import type { 
  AnalysisRequest, 
  Material, 
  BoundaryCondition, 
  Load 
} from '../types';

export interface ValidationError {
  field: string;
  message: string;
}

export function validateAnalysisRequest(
  request: unknown
): ValidationError[] {
  const errors: ValidationError[] = [];
  
  if (!request || typeof request !== 'object') {
    errors.push({ field: 'request', message: 'Request must be an object' });
    return errors;
  }
  
  const req = request as Record<string, unknown>;
  
  // Validate mesh
  if (!req.mesh) {
    errors.push({ field: 'mesh', message: 'Mesh is required' });
  } else if (typeof req.mesh === 'object') {
    const mesh = req.mesh as Record<string, unknown>;
    if (!mesh.format) {
      errors.push({ field: 'mesh.format', message: 'Mesh format is required' });
    }
    if (!mesh.data && !mesh.file_url) {
      errors.push({ 
        field: 'mesh', 
        message: 'Either mesh data or file_url is required' 
      });
    }
  }
  
  // Validate materials
  if (!req.materials) {
    errors.push({ field: 'materials', message: 'Materials are required' });
  } else if (!Array.isArray(req.materials)) {
    errors.push({ field: 'materials', message: 'Materials must be an array' });
  } else if (req.materials.length === 0) {
    errors.push({ 
      field: 'materials', 
      message: 'At least one material is required' 
    });
  } else {
    (req.materials as unknown[]).forEach((mat, idx) => {
      const matErrors = validateMaterial(mat);
      matErrors.forEach(e => {
        errors.push({ 
          field: `materials[${idx}].${e.field}`, 
          message: e.message 
        });
      });
    });
  }
  
  // Validate boundary conditions
  if (!req.boundary_conditions) {
    errors.push({ 
      field: 'boundary_conditions', 
      message: 'Boundary conditions are required' 
    });
  } else if (!Array.isArray(req.boundary_conditions)) {
    errors.push({ 
      field: 'boundary_conditions', 
      message: 'Boundary conditions must be an array' 
    });
  } else if (req.boundary_conditions.length === 0) {
    errors.push({ 
      field: 'boundary_conditions', 
      message: 'At least one boundary condition is required' 
    });
  }
  
  return errors;
}

export function validateMaterial(material: unknown): ValidationError[] {
  const errors: ValidationError[] = [];
  
  if (!material || typeof material !== 'object') {
    errors.push({ field: 'material', message: 'Material must be an object' });
    return errors;
  }
  
  const mat = material as Record<string, unknown>;
  
  if (!mat.id && mat.id !== 0) {
    errors.push({ field: 'id', message: 'Material ID is required' });
  }
  
  if (!mat.properties || typeof mat.properties !== 'object') {
    errors.push({ field: 'properties', message: 'Material properties are required' });
    return errors;
  }
  
  const props = mat.properties as Record<string, unknown>;
  
  switch (props.type) {
    case 'linear_elastic':
      if (typeof props.E !== 'number' || props.E <= 0) {
        errors.push({ 
          field: 'properties.E', 
          message: "Young's modulus must be positive" 
        });
      }
      if (typeof props.nu !== 'number' || props.nu <= -1 || props.nu >= 0.5) {
        errors.push({ 
          field: 'properties.nu', 
          message: 'Poisson ratio must be in (-1, 0.5)' 
        });
      }
      break;
      
    case 'orthotropic':
      ['E1', 'E2', 'E3', 'G12', 'G13', 'G23'].forEach(prop => {
        if (typeof props[prop] !== 'number' || (props[prop] as number) <= 0) {
          errors.push({ 
            field: `properties.${prop}`, 
            message: `${prop} must be positive` 
          });
        }
      });
      break;
      
    default:
      if (!props.type) {
        errors.push({ 
          field: 'properties.type', 
          message: 'Material type is required' 
        });
      }
  }
  
  return errors;
}

export function validateBoundaryCondition(bc: unknown): ValidationError[] {
  const errors: ValidationError[] = [];
  
  if (!bc || typeof bc !== 'object') {
    errors.push({ field: 'bc', message: 'Boundary condition must be an object' });
    return errors;
  }
  
  const condition = bc as Record<string, unknown>;
  
  if (!condition.type) {
    errors.push({ field: 'type', message: 'BC type is required' });
  }
  
  if (condition.boundary_id === undefined) {
    errors.push({ field: 'boundary_id', message: 'Boundary ID is required' });
  }
  
  return errors;
}

export function validateLoad(load: unknown): ValidationError[] {
  const errors: ValidationError[] = [];
  
  if (!load || typeof load !== 'object') {
    errors.push({ field: 'load', message: 'Load must be an object' });
    return errors;
  }
  
  const l = load as Record<string, unknown>;
  
  if (!l.type) {
    errors.push({ field: 'type', message: 'Load type is required' });
  }
  
  return errors;
}
