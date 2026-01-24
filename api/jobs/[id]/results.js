/**
 * FEA API Gateway - Job Results Endpoint
 * 
 * GET /api/jobs/{id}/results - Get analysis results
 */

const COMPUTE_SERVER = process.env.COMPUTE_SERVER_URL;

/**
 * Set CORS headers
 */
function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

/**
 * Validate job ID format
 */
function isValidJobId(id) {
  return /^[a-zA-Z0-9_-]+$/.test(id);
}

export default async function handler(req, res) {
  setCorsHeaders(res);
  
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // Only allow GET
  if (req.method !== 'GET') {
    return res.status(405).json({
      error: 'Method not allowed',
      allowed: ['GET', 'OPTIONS']
    });
  }
  
  const { id } = req.query;
  
  // Validate job ID
  if (!id || !isValidJobId(id)) {
    return res.status(400).json({
      error: 'Invalid job ID'
    });
  }
  
  // Check compute server URL
  if (!COMPUTE_SERVER) {
    return res.status(503).json({
      error: 'Service configuration error'
    });
  }
  
  try {
    const response = await fetch(`${COMPUTE_SERVER}/api/jobs/${id}/results`, {
      method: 'GET',
      signal: AbortSignal.timeout(30000)
    });
    
    const data = await response.json();
    
    // Add gateway metadata
    if (response.status === 200 && data.status === 'completed') {
      data.gateway_timestamp = new Date().toISOString();
    }
    
    return res.status(response.status).json(data);
    
  } catch (error) {
    console.error('Error fetching results:', error);
    
    if (error.name === 'TimeoutError') {
      return res.status(504).json({
        error: 'Gateway timeout'
      });
    }
    
    return res.status(503).json({
      error: 'Compute server unavailable'
    });
  }
}
