/**
 * FEA API Gateway - Job Files Download Endpoint
 * 
 * GET /api/jobs/{id}/files/{filename} - Download output files
 */

const COMPUTE_SERVER = process.env.COMPUTE_SERVER_URL;

export const config = {
  api: {
    responseLimit: '100mb'
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

/**
 * Validate filename (prevent directory traversal)
 */
function isValidFilename(filename) {
  // Must not contain .. or start with /
  if (filename.includes('..') || filename.startsWith('/')) {
    return false;
  }
  // Allow common output files
  return /^[a-zA-Z0-9_.-]+\.(vtu|vtk|csv|json|txt)$/.test(filename);
}

/**
 * Get content type for file extension
 */
function getContentType(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  const contentTypes = {
    'vtu': 'application/xml',
    'vtk': 'application/xml',
    'csv': 'text/csv',
    'json': 'application/json',
    'txt': 'text/plain'
  };
  return contentTypes[ext] || 'application/octet-stream';
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
      error: 'Method not allowed'
    });
  }
  
  const { id, filename } = req.query;
  
  // Validate job ID
  if (!id || !isValidJobId(id)) {
    return res.status(400).json({
      error: 'Invalid job ID'
    });
  }
  
  // Validate filename
  if (!filename || !isValidFilename(filename)) {
    return res.status(400).json({
      error: 'Invalid filename',
      message: 'Filename must be alphanumeric with allowed extensions: .vtu, .vtk, .csv, .json, .txt'
    });
  }
  
  // Check compute server URL
  if (!COMPUTE_SERVER) {
    return res.status(503).json({
      error: 'Service configuration error'
    });
  }
  
  try {
    const response = await fetch(`${COMPUTE_SERVER}/api/jobs/${id}/files/${filename}`, {
      method: 'GET',
      signal: AbortSignal.timeout(60000)
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      return res.status(response.status).json(errorData);
    }
    
    // Get file content
    const fileBuffer = await response.arrayBuffer();
    
    // Set response headers
    const contentType = getContentType(filename);
    res.setHeader('Content-Type', contentType);
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('Content-Length', fileBuffer.byteLength);
    
    // Send file
    return res.status(200).send(Buffer.from(fileBuffer));
    
  } catch (error) {
    console.error('Error downloading file:', error);
    
    if (error.name === 'TimeoutError') {
      return res.status(504).json({
        error: 'Gateway timeout',
        message: 'File download timed out'
      });
    }
    
    return res.status(503).json({
      error: 'Compute server unavailable'
    });
  }
}
