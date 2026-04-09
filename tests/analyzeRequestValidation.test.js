import test from 'node:test';
import assert from 'node:assert/strict';
import { validateAnalyzeRequest, validateMeshObject } from '../lib/analyzeRequestValidation.js';

test('rejects sphere target with actionable message', () => {
  const errs = validateAnalyzeRequest({
    mesh: { type: 'box', min: [0, 0, 0], max: [1, 1, 1] },
    boundary_conditions: [
      { type: 'fixed', target: { type: 'sphere', center: [0, 0, 0], radius: 1 } }
    ]
  });
  assert.ok(errs.some((e) => e.includes('sphere') && e.includes('not supported')));
});

test('accepts box target', () => {
  const errs = validateAnalyzeRequest({
    mesh: { type: 'box', min: [0, 0, 0], max: [1, 1, 1] },
    boundary_conditions: [
      { type: 'fixed', target: { type: 'box', min: [0, 0, 0], max: [1, 0, 1] } }
    ]
  });
  assert.equal(errs.length, 0);
});

test('file mesh requires format msh when data present', () => {
  const errs = validateMeshObject({ type: 'file', data: '$MeshFormat\n' });
  assert.ok(errs.some((e) => e.includes('mesh.format')));
});

test('inline material requires properties', () => {
  const errs = validateAnalyzeRequest({
    mesh: { type: 'box', min: [0, 0, 0], max: [1, 1, 1] },
    boundary_conditions: [
      { type: 'fixed', target: { type: 'box', min: [0, 0, 0], max: [0, 1, 1] } }
    ],
    materials: { default: { model: 'isotropic_elastic' } }
  });
  assert.ok(errs.some((e) => e.includes('properties')));
});
