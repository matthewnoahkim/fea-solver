# Validation Test: Thick-Walled Pressurized Cylinder

## Problem Description
A thick-walled cylinder with inner radius a, outer radius b, under internal pressure p.

## Analytical Solution (Lamé Equations)
- Radial stress: σ_r = (a²p/(b²-a²)) × (1 - b²/r²)
- Hoop stress: σ_θ = (a²p/(b²-a²)) × (1 + b²/r²)
- Radial displacement: u_r = (a²p/E(b²-a²)) × ((1-ν)r + (1+ν)b²/r)

## Test Parameters
- Inner radius: a = 0.1 m
- Outer radius: b = 0.2 m
- Material: Steel (E = 200 GPa, ν = 0.3)
- Internal pressure: p = 10 MPa

## Expected Results (at inner surface r = a)
- σ_r = -10 MPa (pressure)
- σ_θ = 16.67 MPa
- u_r = 0.0125 mm

## Acceptance Criteria
- Stress error < 2%
- Displacement error < 2%
