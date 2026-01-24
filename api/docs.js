/**
 * FEA API Gateway - Documentation Endpoint
 * 
 * GET /api/docs - Redirect to documentation
 */

import { readFileSync } from 'fs';
import { join } from 'path';

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Redirect to docs page
  res.redirect(302, '/docs');
}
