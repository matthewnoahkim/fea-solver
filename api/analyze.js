/**
 * FEA API Gateway - Analysis Submission Endpoint
 * 
 * POST /api/analyze
 * 
 * Submits a new analysis job to the compute server.
 */

import { validateAnalyzeRequest } from '../lib/analyzeRequestValidation.js';

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
  const validationErrors = validateAnalyzeRequest(req.body);
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
        message: 'Compute server request timed out',
        details: [
          'The gateway waited ~55s for the compute server to accept the job. If you received a job_id earlier, keep polling GET /api/jobs/{id}.',
          'For large meshes, ensure the compute tier returns 202 quickly after enqueue; see docs/openapi.yaml (Gateway timeouts).'
        ]
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
