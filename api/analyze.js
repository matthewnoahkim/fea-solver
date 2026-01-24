/**
 * FEA API Gateway - Analysis Submission Endpoint
 * 
 * POST /api/analyze
 * 
 * Submits a new analysis job to the compute server.
 */

const COMPUTE_SERVER = process.env.COMPUTE_SERVER_URL;

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '50mb'
    }
  },
  maxDuration: 60
};

/**
 * Set CORS headers for the response
 */
function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
}

/**
 * Validate the analysis request
 */
function validateRequest(body) {
  const errors = [];
  
  if (!body || typeof body !== 'object') {
    errors.push('Request body must be a JSON object');
    return errors;
  }
  
  // Check required fields
  if (!body.mesh) {
    errors.push('Missing required field: mesh');
  } else {
    if (!body.mesh.type) {
      errors.push('Missing required field: mesh.type');
    }
    if (body.mesh.type === 'box') {
      if (!body.mesh.min || !Array.isArray(body.mesh.min) || body.mesh.min.length !== 3) {
        errors.push('Box mesh requires min: [x, y, z]');
      }
      if (!body.mesh.max || !Array.isArray(body.mesh.max) || body.mesh.max.length !== 3) {
        errors.push('Box mesh requires max: [x, y, z]');
      }
    }
    if (body.mesh.type === 'file' && !body.mesh.data && !body.mesh.path && !body.mesh.url) {
      errors.push('File mesh requires data, path, or url');
    }
  }
  
  if (!body.boundary_conditions) {
    errors.push('Missing required field: boundary_conditions');
  } else if (!Array.isArray(body.boundary_conditions)) {
    errors.push('boundary_conditions must be an array');
  } else if (body.boundary_conditions.length === 0) {
    errors.push('boundary_conditions must not be empty');
  } else {
    // Validate each BC has a type
    body.boundary_conditions.forEach((bc, i) => {
      if (!bc.type) {
        errors.push(`boundary_conditions[${i}] missing type`);
      }
    });
  }
  
  // Validate loads if present
  if (body.loads && !Array.isArray(body.loads)) {
    errors.push('loads must be an array');
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
  const validationErrors = validateRequest(req.body);
  if (validationErrors.length > 0) {
    return res.status(400).json({
      error: 'Validation failed',
      details: validationErrors
    });
  }
  
  // Check compute server URL is configured
  if (!COMPUTE_SERVER) {
    console.error('COMPUTE_SERVER_URL environment variable not set');
    return res.status(503).json({
      error: 'Service configuration error',
      message: 'Compute server URL not configured'
    });
  }
  
  try {
    // Forward request to compute server
    const response = await fetch(`${COMPUTE_SERVER}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(req.body),
      // Set timeout
      signal: AbortSignal.timeout(55000)
    });
    
    const data = await response.json();
    
    // Forward response with same status code
    return res.status(response.status).json(data);
    
  } catch (error) {
    console.error('Compute server error:', error);
    
    // Handle timeout
    if (error.name === 'TimeoutError' || error.name === 'AbortError') {
      return res.status(504).json({
        error: 'Gateway timeout',
        message: 'Compute server request timed out'
      });
    }
    
    // Handle connection errors
    if (error.cause?.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'Compute server unavailable',
        message: 'Cannot connect to compute server'
      });
    }
    
    return res.status(503).json({
      error: 'Compute server error',
      message: 'Please try again later'
    });
  }
}
