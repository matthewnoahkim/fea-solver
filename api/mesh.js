/**
 * FEA API Gateway - Mesh Quality Analysis Endpoint
 * 
 * POST /api/mesh/quality - Analyze mesh quality
 */

const COMPUTE_SERVER = process.env.COMPUTE_SERVER_URL;

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '50mb'
    }
  },
  maxDuration: 30
};

/**
 * Set CORS headers
 */
function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

/**
 * Validate mesh specification
 */
function validateMesh(body) {
  const errors = [];
  
  if (!body || typeof body !== 'object') {
    errors.push('Request body must be a JSON object');
    return errors;
  }
  
  if (!body.mesh) {
    errors.push('Missing required field: mesh');
    return errors;
  }
  
  const mesh = body.mesh;
  
  if (!mesh.type) {
    errors.push('Missing mesh.type');
  } else if (!['box', 'cylinder', 'file'].includes(mesh.type)) {
    errors.push('mesh.type must be one of: box, cylinder, file');
  }
  
  if (mesh.type === 'box') {
    if (!mesh.min || !Array.isArray(mesh.min) || mesh.min.length !== 3) {
      errors.push('box mesh requires min: [x, y, z]');
    }
    if (!mesh.max || !Array.isArray(mesh.max) || mesh.max.length !== 3) {
      errors.push('box mesh requires max: [x, y, z]');
    }
  }
  
  if (mesh.type === 'cylinder') {
    if (typeof mesh.radius !== 'number' || mesh.radius <= 0) {
      errors.push('cylinder mesh requires positive radius');
    }
    if (typeof mesh.height !== 'number' || mesh.height <= 0) {
      errors.push('cylinder mesh requires positive height');
    }
  }
  
  if (mesh.type === 'file') {
    if (!mesh.data && !mesh.path && !mesh.url) {
      errors.push('file mesh requires data, path, or url');
    }
  }
  
  return errors;
}

export default async function handler(req, res) {
  setCorsHeaders(res);
  
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({
      error: 'Method not allowed',
      allowed: ['POST', 'OPTIONS']
    });
  }
  
  // Validate request
  const errors = validateMesh(req.body);
  if (errors.length > 0) {
    return res.status(400).json({
      error: 'Validation failed',
      details: errors
    });
  }
  
  // Check compute server URL
  if (!COMPUTE_SERVER) {
    return res.status(503).json({
      error: 'Service configuration error',
      message: 'Compute server URL not configured'
    });
  }
  
  try {
    const response = await fetch(`${COMPUTE_SERVER}/api/mesh/quality`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(req.body),
      signal: AbortSignal.timeout(25000)
    });
    
    const data = await response.json();
    return res.status(response.status).json(data);
    
  } catch (error) {
    console.error('Mesh quality analysis error:', error);
    
    if (error.name === 'TimeoutError') {
      return res.status(504).json({
        error: 'Gateway timeout',
        message: 'Mesh analysis timed out - mesh may be too large'
      });
    }
    
    return res.status(503).json({
      error: 'Compute server unavailable'
    });
  }
}
