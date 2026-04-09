/**
 * FEA API Gateway - Job Status Endpoint
 * 
 * GET /api/jobs/{id} - Get job status
 * DELETE /api/jobs/{id} - Cancel job
 */

const COMPUTE_SERVER = process.env.COMPUTE_SERVER_URL;

/**
 * Set CORS headers
 */
function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

/**
 * Validate job ID format
 */
function isValidJobId(id) {
  // Allow alphanumeric, underscores, and hyphens
  return /^[a-zA-Z0-9_-]+$/.test(id);
}

export default async function handler(req, res) {
  setCorsHeaders(res);
  
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // Extract job ID from query
  const { id } = req.query;
  
  // Validate job ID
  if (!id || !isValidJobId(id)) {
    return res.status(400).json({
      error: 'Invalid job ID',
      details: ['Job ID must contain only alphanumeric characters, underscores, and hyphens']
    });
  }
  
  // Check compute server URL
  if (!COMPUTE_SERVER) {
    return res.status(503).json({
      error: 'Service configuration error'
    });
  }
  
  try {
    if (req.method === 'GET') {
      // Get job status
      const response = await fetch(`${COMPUTE_SERVER}/api/jobs/${id}`, {
        method: 'GET',
        signal: AbortSignal.timeout(10000)
      });
      
      const data = await response.json();
      if (
        typeof data.retry_after_seconds === 'number' &&
        data.retry_after_seconds >= 0 &&
        Number.isFinite(data.retry_after_seconds)
      ) {
        const ra = Math.min(Math.ceil(data.retry_after_seconds), 86400);
        res.setHeader('Retry-After', String(ra));
      }
      return res.status(response.status).json(data);

    } else if (req.method === 'DELETE') {
      // Cancel job
      const response = await fetch(`${COMPUTE_SERVER}/api/jobs/${id}`, {
        method: 'DELETE',
        signal: AbortSignal.timeout(10000)
      });

      const data = await response.json();
      return res.status(response.status).json(data);
      
    } else {
      return res.status(405).json({
        error: 'Method not allowed',
        allowed: ['GET', 'DELETE', 'OPTIONS']
      });
    }
    
  } catch (error) {
    console.error('Compute server error:', error);
    
    if (error.name === 'TimeoutError') {
      return res.status(504).json({
        error: 'Gateway timeout',
        details: [
          'Job status request timed out at the gateway (~10s). Retry GET /api/jobs/{id}; the compute server may be slow or unreachable.'
        ]
      });
    }

    return res.status(503).json({
      error: 'Compute server unavailable',
      details: ['Could not reach the compute server for job status']
    });
  }
}
