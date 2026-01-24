/**
 * @file linear_elastic.cc
 * @brief Implementation of linear elastic material utilities
 */

#include "linear_elastic.h"
#include <cmath>

namespace FEA {

// ============================================================================
// LinearElasticMaterial Implementation
// ============================================================================

LinearElasticMaterial::LinearElasticMaterial(const IsotropicElasticProperties& props)
    : E_(props.youngs_modulus),
      nu_(props.poissons_ratio),
      rho_(props.density),
      alpha_(props.thermal_expansion_coeff)
{
    // Compute derived constants
    lambda_ = props.lambda();
    mu_ = props.mu();
    K_ = props.bulk_modulus();
    
    // Build elasticity tensor
    elasticity_tensor_ = build_isotropic_elasticity_tensor(lambda_, mu_);
}

SymmetricTensor<2, 3> LinearElasticMaterial::compute_stress(
    const SymmetricTensor<2, 3>& strain) const
{
    // σ = C : ε
    return elasticity_tensor_ * strain;
}

SymmetricTensor<2, 3> LinearElasticMaterial::compute_stress_with_thermal(
    const SymmetricTensor<2, 3>& strain,
    double delta_temperature) const
{
    // Thermal strain: ε_th = α * ΔT * I
    SymmetricTensor<2, 3> thermal_strain;
    for (unsigned int i = 0; i < 3; ++i) {
        thermal_strain[i][i] = alpha_ * delta_temperature;
    }
    
    // Mechanical strain: ε_mech = ε_total - ε_th
    SymmetricTensor<2, 3> mechanical_strain = strain - thermal_strain;
    
    // Stress from mechanical strain only
    return elasticity_tensor_ * mechanical_strain;
}

SymmetricTensor<2, 3> LinearElasticMaterial::get_thermal_expansion_tensor() const
{
    SymmetricTensor<2, 3> alpha_tensor;
    alpha_tensor = 0;
    for (unsigned int i = 0; i < 3; ++i) {
        alpha_tensor[i][i] = alpha_;
    }
    return alpha_tensor;
}

double LinearElasticMaterial::compute_strain_energy(
    const SymmetricTensor<2, 3>& strain) const
{
    // W = 0.5 * σ : ε = 0.5 * ε : C : ε
    SymmetricTensor<2, 3> stress = compute_stress(strain);
    return 0.5 * (stress * strain);  // Double contraction
}

// ============================================================================
// Free Functions
// ============================================================================

SymmetricTensor<4, 3> build_isotropic_elasticity_tensor(double lambda, double mu)
{
    SymmetricTensor<4, 3> C;
    
    // C_ijkl = λ*δ_ij*δ_kl + μ*(δ_ik*δ_jl + δ_il*δ_jk)
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {
                for (unsigned int l = 0; l < 3; ++l) {
                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    double delta_kl = (k == l) ? 1.0 : 0.0;
                    double delta_ik = (i == k) ? 1.0 : 0.0;
                    double delta_jl = (j == l) ? 1.0 : 0.0;
                    double delta_il = (i == l) ? 1.0 : 0.0;
                    double delta_jk = (j == k) ? 1.0 : 0.0;
                    
                    C[i][j][k][l] = lambda * delta_ij * delta_kl
                                  + mu * (delta_ik * delta_jl + delta_il * delta_jk);
                }
            }
        }
    }
    
    return C;
}

SymmetricTensor<4, 3> invert_elasticity_tensor(const SymmetricTensor<4, 3>& C)
{
    // For isotropic materials: S_ijkl = -λ/(2μ(3λ+2μ))*δ_ij*δ_kl + 1/(4μ)*(δ_ik*δ_jl + δ_il*δ_jk)
    // First extract λ and μ from C
    
    // C_1111 = λ + 2μ, C_1122 = λ, C_1212 = μ
    double lambda_plus_2mu = C[0][0][0][0];
    double lambda = C[0][0][1][1];
    double mu = C[0][1][0][1];
    
    // Build compliance tensor
    SymmetricTensor<4, 3> S;
    
    double coeff1 = -lambda / (2.0 * mu * (3.0 * lambda + 2.0 * mu));
    double coeff2 = 1.0 / (4.0 * mu);
    
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {
                for (unsigned int l = 0; l < 3; ++l) {
                    double delta_ij = (i == j) ? 1.0 : 0.0;
                    double delta_kl = (k == l) ? 1.0 : 0.0;
                    double delta_ik = (i == k) ? 1.0 : 0.0;
                    double delta_jl = (j == l) ? 1.0 : 0.0;
                    double delta_il = (i == l) ? 1.0 : 0.0;
                    double delta_jk = (j == k) ? 1.0 : 0.0;
                    
                    S[i][j][k][l] = coeff1 * delta_ij * delta_kl
                                  + coeff2 * (delta_ik * delta_jl + delta_il * delta_jk);
                }
            }
        }
    }
    
    return S;
}

SymmetricTensor<4, 2> build_plane_stress_tensor(double E, double nu)
{
    // For plane stress: σ_33 = σ_13 = σ_23 = 0
    // Reduced constitutive matrix:
    // [σ11]   E     [1   ν   0   ] [ε11]
    // [σ22] = ---- * [ν   1   0   ] [ε22]
    // [σ12]   1-ν²  [0   0  (1-ν)/2] [2ε12]
    
    SymmetricTensor<4, 2> C;
    C = 0;
    
    double factor = E / (1.0 - nu * nu);
    
    C[0][0][0][0] = factor;
    C[1][1][1][1] = factor;
    C[0][0][1][1] = C[1][1][0][0] = factor * nu;
    
    double shear_factor = factor * (1.0 - nu) / 2.0;
    C[0][1][0][1] = C[1][0][1][0] = C[0][1][1][0] = C[1][0][0][1] = shear_factor / 2.0;
    
    return C;
}

SymmetricTensor<4, 2> build_plane_strain_tensor(double E, double nu)
{
    // For plane strain: ε_33 = ε_13 = ε_23 = 0
    // Same as 3D but in-plane only
    // λ_eff = λ, μ_eff = μ
    
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));
    
    SymmetricTensor<4, 2> C;
    C = 0;
    
    // C_1111 = C_2222 = λ + 2μ
    C[0][0][0][0] = lambda + 2.0 * mu;
    C[1][1][1][1] = lambda + 2.0 * mu;
    
    // C_1122 = λ
    C[0][0][1][1] = C[1][1][0][0] = lambda;
    
    // C_1212 = μ (but tensor storage factor)
    C[0][1][0][1] = C[1][0][1][0] = C[0][1][1][0] = C[1][0][0][1] = mu / 2.0;
    
    return C;
}

} // namespace FEA
