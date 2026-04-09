/**
 * FEA API Gateway - Mesh Quality Analysis Endpoint
 * 
 * POST /api/mesh/quality - Analyze mesh quality
 */

import { validateMeshObject } from '../lib/analyzeRequestValidation.js';

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
 * Validate mesh specification (same rules as analyze mesh subset).
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

  errors.push(...validateMeshObject(body.mesh));
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
