/**
 * FEA API Gateway - Job Results Endpoint
 * 
 * GET /api/jobs/{id}/results - Get analysis results
 */

const COMPUTE_SERVER = process.env.COMPUTE_SERVER_URL;

/** SI outputs for completed jobs — independent of request units.type (see docs/openapi.yaml). */
const COMPLETED_RESULTS_METADATA = {
  results_schema_version: '1',
  results_units: {
    length: 'm',
    displacement: 'm',
    stress: 'Pa',
    force: 'N'
  }
};

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
    const response = await fetch(`${COMPUTE_SERVER}/api/jobs/${id}/results`, {
      method: 'GET',
      signal: AbortSignal.timeout(30000)
    });
    
    const data = await response.json();

    if (response.status === 200 && data && data.status === 'completed') {
      Object.assign(data, COMPLETED_RESULTS_METADATA, {
        gateway_timestamp: new Date().toISOString()
      });
    }

    return res.status(response.status).json(data);
    
  } catch (error) {
    console.error('Error fetching results:', error);
    
    if (error.name === 'TimeoutError') {
      return res.status(504).json({
        error: 'Gateway timeout',
        details: [
          'Results fetch timed out at the gateway (~30s). The job may still be running — poll GET /api/jobs/{id} and retry results when status is completed.'
        ]
      });
    }

    return res.status(503).json({
      error: 'Compute server unavailable',
      details: ['Could not reach the compute server for results']
    });
  }
}
