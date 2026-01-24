/**
 * FEA API Gateway - Health Check Endpoint
 * 
 * GET /api/health - Check gateway and compute server health
 */

const COMPUTE_SERVER = process.env.COMPUTE_SERVER_URL;

/**
 * Set CORS headers
 */
function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

export default async function handler(req, res) {
  setCorsHeaders(res);
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const response = {
    gateway: {
      status: 'healthy',
      version: '1.0.0',
      timestamp: new Date().toISOString()
    },
    compute_server: {
      status: 'unknown',
      url: COMPUTE_SERVER ? 'configured' : 'not_configured'
    }
  };
  
  // Check compute server health
  if (COMPUTE_SERVER) {
    try {
      const serverResponse = await fetch(`${COMPUTE_SERVER}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      
      if (serverResponse.ok) {
        const serverHealth = await serverResponse.json();
        response.compute_server = {
          status: 'healthy',
          ...serverHealth
        };
      } else {
        response.compute_server.status = 'unhealthy';
        response.compute_server.error = `HTTP ${serverResponse.status}`;
      }
      
    } catch (error) {
      response.compute_server.status = 'unavailable';
      response.compute_server.error = error.name === 'TimeoutError' 
        ? 'Connection timeout' 
        : 'Connection failed';
    }
  }
  
  // Determine overall status
  const overallHealthy = response.gateway.status === 'healthy' && 
                         response.compute_server.status === 'healthy';
  
  return res.status(overallHealthy ? 200 : 503).json(response);
}
