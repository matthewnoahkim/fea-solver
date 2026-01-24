/**
 * FEA API Gateway - Materials Endpoint
 * 
 * GET /api/materials - List available materials
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

/**
 * Default materials list (fallback if compute server unavailable)
 */
const DEFAULT_MATERIALS = {
  materials: [
    {
      id: 'steel_structural',
      name: 'Structural Steel',
      model: 'isotropic_elastic',
      properties: {
        youngs_modulus: 200e9,
        poissons_ratio: 0.3,
        density: 7850,
        thermal_expansion: 12e-6
      },
      yield_strength: 250e6,
      ultimate_strength: 400e6
    },
    {
      id: 'aluminum_6061_t6',
      name: 'Aluminum 6061-T6',
      model: 'isotropic_elastic',
      properties: {
        youngs_modulus: 68.9e9,
        poissons_ratio: 0.33,
        density: 2700,
        thermal_expansion: 23.6e-6
      },
      yield_strength: 276e6,
      ultimate_strength: 310e6
    },
    {
      id: 'titanium_ti6al4v',
      name: 'Titanium Ti-6Al-4V',
      model: 'isotropic_elastic',
      properties: {
        youngs_modulus: 113.8e9,
        poissons_ratio: 0.342,
        density: 4430,
        thermal_expansion: 8.6e-6
      },
      yield_strength: 880e6,
      ultimate_strength: 950e6
    },
    {
      id: 'stainless_304',
      name: 'Stainless Steel 304',
      model: 'isotropic_elastic',
      properties: {
        youngs_modulus: 193e9,
        poissons_ratio: 0.29,
        density: 8000,
        thermal_expansion: 17.3e-6
      },
      yield_strength: 215e6,
      ultimate_strength: 505e6
    },
    {
      id: 'copper_annealed',
      name: 'Copper (Annealed)',
      model: 'isotropic_elastic',
      properties: {
        youngs_modulus: 110e9,
        poissons_ratio: 0.343,
        density: 8960,
        thermal_expansion: 16.5e-6
      },
      yield_strength: 70e6,
      ultimate_strength: 220e6
    }
  ]
};

export default async function handler(req, res) {
  setCorsHeaders(res);
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Try to get materials from compute server
  if (COMPUTE_SERVER) {
    try {
      const response = await fetch(`${COMPUTE_SERVER}/api/materials`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        const data = await response.json();
        return res.status(200).json(data);
      }
      
    } catch (error) {
      console.warn('Failed to fetch materials from compute server:', error.message);
      // Fall through to default materials
    }
  }
  
  // Return default materials list
  return res.status(200).json({
    ...DEFAULT_MATERIALS,
    source: 'gateway_fallback'
  });
}
