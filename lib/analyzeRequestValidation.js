/**
 * Shared validation for POST /api/analyze (and mesh.file rules for /api/mesh/quality).
 * Aligns with docs/openapi.yaml — compute server should return the same structured errors when it rejects payloads.
 */

const SUPPORTED_TARGET_TYPES = new Set(['boundary_id', 'point', 'box']);

/**
 * @param {unknown} mesh
 * @returns {string[]}
 */
export function validateMeshObject(mesh) {
  const errors = [];
  if (!mesh || typeof mesh !== 'object') {
    errors.push('mesh must be an object');
    return errors;
  }
  if (!mesh.type) {
    errors.push('Missing required field: mesh.type');
    return errors;
  }

  if (mesh.type === 'box') {
    if (!mesh.min || !Array.isArray(mesh.min) || mesh.min.length !== 3) {
      errors.push('Box mesh requires min: [x, y, z]');
    }
    if (!mesh.max || !Array.isArray(mesh.max) || mesh.max.length !== 3) {
      errors.push('Box mesh requires max: [x, y, z]');
    }
  } else if (mesh.type === 'cylinder') {
    if (typeof mesh.radius !== 'number' || mesh.radius <= 0) {
      errors.push('cylinder mesh requires positive radius');
    }
    if (typeof mesh.height !== 'number' || mesh.height <= 0) {
      errors.push('cylinder mesh requires positive height');
    }
  } else if (mesh.type === 'file') {
    if (!mesh.data && !mesh.path && !mesh.url) {
      errors.push('File mesh requires data, path, or url');
    }
    validateFileMeshFields(mesh, errors);
  } else {
    errors.push(`mesh.type must be one of: box, cylinder, file (received "${mesh.type}")`);
  }

  return errors;
}

/**
 * @param {Record<string, unknown>} mesh
 * @param {string[]} errors
 */
function validateFileMeshFields(mesh, errors) {
  const encoding = mesh.encoding === undefined ? 'utf8' : mesh.encoding;
  if (encoding !== 'utf8' && encoding !== 'base64') {
    errors.push(
      'mesh.encoding must be "utf8" (default: entire .msh file as UTF-8 text in mesh.data) or "base64" (standard base64 of the .msh file bytes)'
    );
  }

  if (mesh.data != null) {
    if (typeof mesh.data !== 'string') {
      errors.push(
        'mesh.data must be a string: UTF-8 .msh text when mesh.encoding is utf8 (default), or base64 when mesh.encoding is base64'
      );
    }
    if (!mesh.format || mesh.format !== 'msh') {
      errors.push('When mesh.data is set, mesh.format must be "msh"');
    }
  }

  if (mesh.format != null && mesh.format !== 'msh') {
    errors.push('mesh.format must be "msh" for the documented file mesh contract');
  }
}

/**
 * @param {unknown} target
 * @param {string} pathLabel e.g. boundary_conditions[0].target
 * @param {string[]} errors
 */
function validateTarget(target, pathLabel, errors) {
  if (!target || typeof target !== 'object') {
    errors.push(`${pathLabel} is required and must be an object`);
    return;
  }
  const type = target.type;
  if (type === 'sphere') {
    errors.push(
      `${pathLabel}: target type "sphere" is not supported. Use "box", "point", or "boundary_id", or omit geometric targets and use boundary_id with Gmsh physical surface tags.`
    );
    return;
  }
  if (typeof type !== 'string' || !SUPPORTED_TARGET_TYPES.has(type)) {
    errors.push(
      `${pathLabel}: unknown or unsupported target.type "${type}". Supported: boundary_id, point, box`
    );
    return;
  }

  if (type === 'boundary_id') {
    if (typeof target.id !== 'number' || !Number.isInteger(target.id)) {
      errors.push(
        `${pathLabel}: boundary_id target requires integer JSON number "id" (Gmsh physical tag on boundary surfaces for file meshes), not a string`
      );
    }
  } else if (type === 'point') {
    const loc = target.location;
    if (!Array.isArray(loc) || loc.length !== 3 || loc.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
      errors.push(`${pathLabel}: point target requires "location": [x, y, z] with finite numbers`);
    }
  } else if (type === 'box') {
    const { min, max } = target;
    if (!Array.isArray(min) || min.length !== 3 || min.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
      errors.push(`${pathLabel}: box target requires "min": [x, y, z] with finite numbers`);
    }
    if (!Array.isArray(max) || max.length !== 3 || max.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
      errors.push(`${pathLabel}: box target requires "max": [x, y, z] with finite numbers`);
    }
  }
}

/**
 * @param {unknown} body
 * @returns {string[]}
 */
export function validateAnalyzeRequest(body) {
  const errors = [];

  if (!body || typeof body !== 'object') {
    errors.push('Request body must be a JSON object');
    return errors;
  }

  errors.push(...validateMeshObject(body.mesh));
  if (!body.mesh || typeof body.mesh !== 'object') {
    return errors;
  }

  if (!body.boundary_conditions) {
    errors.push('Missing required field: boundary_conditions');
  } else if (!Array.isArray(body.boundary_conditions)) {
    errors.push('boundary_conditions must be an array');
  } else if (body.boundary_conditions.length === 0) {
    errors.push('boundary_conditions must not be empty');
  } else {
    body.boundary_conditions.forEach((bc, i) => {
      if (!bc || typeof bc !== 'object') {
        errors.push(`boundary_conditions[${i}] must be an object`);
        return;
      }
      if (!bc.type) {
        errors.push(`boundary_conditions[${i}] missing type`);
      }
      if (!bc.target) {
        errors.push(`boundary_conditions[${i}] missing target (fixed, displacement, symmetry, etc. require a target)`);
      } else {
        validateTarget(bc.target, `boundary_conditions[${i}].target`, errors);
      }
    });
  }

  if (body.loads && !Array.isArray(body.loads)) {
    errors.push('loads must be an array');
  } else if (Array.isArray(body.loads)) {
    body.loads.forEach((load, i) => {
      if (!load || typeof load !== 'object') {
        errors.push(`loads[${i}] must be an object`);
        return;
      }
      const t = load.type;
      if (load.target) {
        validateTarget(load.target, `loads[${i}].target`, errors);
      }
      if (t === 'pressure' || t === 'surface_force') {
        if (!load.target) {
          errors.push(`loads[${i}] (${t}) requires "target"`);
        }
      }
    });
  }

  validateMaterials(body.materials, errors);

  return errors;
}

/**
 * @param {unknown} materials
 * @param {string[]} errors
 */
function validateMaterials(materials, errors) {
  if (materials == null) return;
  if (typeof materials !== 'object' || Array.isArray(materials)) {
    errors.push('materials must be an object');
    return;
  }
  if (materials.default === undefined) return;

  const d = materials.default;
  if (typeof d === 'string') {
    if (!d.trim()) {
      errors.push('materials.default must be a non-empty preset id string when using a string value');
    }
    return;
  }
  if (typeof d !== 'object' || d === null) {
    errors.push(
      'materials.default must be either a preset id string (from GET /api/materials) or an inline material object with model + properties (see OpenAPI schema)'
    );
    return;
  }

  if (!d.model || typeof d.model !== 'string' || !d.model.trim()) {
    errors.push('Inline materials.default requires string "model" (e.g. "isotropic_elastic")');
  }
  const p = d.properties;
  if (!p || typeof p !== 'object') {
    errors.push(
      'Inline materials.default requires object "properties" with youngs_modulus (Pa), poissons_ratio, density (kg/m³)'
    );
    return;
  }
  for (const key of ['youngs_modulus', 'poissons_ratio', 'density']) {
    const v = p[key];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      errors.push(`Inline materials.default.properties.${key} must be a finite number`);
    }
  }
  if (d.yield_strength !== undefined) {
    if (typeof d.yield_strength !== 'number' || !Number.isFinite(d.yield_strength)) {
      errors.push('Inline materials.default.yield_strength must be a finite number (Pa) if provided');
    }
  }
}
