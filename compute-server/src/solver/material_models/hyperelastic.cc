/**
 * @file hyperelastic.cc
 * @brief Implementation of hyperelastic material models
 */

#include "hyperelastic.h"
#include <cmath>
#include <algorithm>
#include <array>

namespace FEA {

// ============================================================================
// Utility Functions
// ============================================================================

SymmetricTensor<2, 3> compute_almansi_strain(const Tensor<2, 3>& F)
{
    SymmetricTensor<2, 3> B = compute_left_cauchy_green(F);
    SymmetricTensor<2, 3> B_inv = invert(B);
    
    SymmetricTensor<2, 3> e;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            e[i][j] = -0.5 * B_inv[i][j];
            if (i == j) e[i][j] += 0.5;
        }
    return e;
}

SymmetricTensor<2, 3> push_forward(
    const SymmetricTensor<2, 3>& A,
    const Tensor<2, 3>& F)
{
    double J = determinant(F);
    
    Tensor<2, 3> temp;
    temp = 0;
    
    // temp = F * A * F^T
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    temp[i][j] += F[i][p] * A[p][q] * F[j][q];
    
    // Scale by 1/J and symmetrize
    SymmetricTensor<2, 3> a;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
            a[i][j] = 0.5 * (temp[i][j] + temp[j][i]) / J;
    
    return a;
}

SymmetricTensor<2, 3> pull_back(
    const SymmetricTensor<2, 3>& a,
    const Tensor<2, 3>& F)
{
    double J = determinant(F);
    Tensor<2, 3> F_inv = invert(F);
    
    Tensor<2, 3> temp;
    temp = 0;
    
    // temp = F^{-1} * a * F^{-T}
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    temp[i][j] += F_inv[i][p] * a[p][q] * F_inv[j][q];
    
    // Scale by J and symmetrize
    SymmetricTensor<2, 3> A;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
            A[i][j] = 0.5 * J * (temp[i][j] + temp[j][i]);
    
    return A;
}

void compute_invariants(
    const SymmetricTensor<2, 3>& A,
    double& I1, double& I2, double& I3)
{
    I1 = trace(A);
    
    // I2 = 0.5 * (tr(A)² - tr(A²))
    double trA2 = 0;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            trA2 += A[i][j] * A[j][i];
    
    I2 = 0.5 * (I1 * I1 - trA2);
    
    I3 = determinant(A);
}

std::array<double, 3> compute_principal_stretches(const Tensor<2, 3>& F)
{
    SymmetricTensor<2, 3> C = compute_right_cauchy_green(F);
    
    // Eigenvalues of C are λ²
    auto eigenvalues = dealii::eigenvalues(C);
    
    std::array<double, 3> stretches;
    for (int i = 0; i < 3; ++i) {
        stretches[i] = std::sqrt(std::max(0.0, eigenvalues[i]));
    }
    
    // Sort in descending order
    std::sort(stretches.begin(), stretches.end(), std::greater<double>());
    
    return stretches;
}

// ============================================================================
// HyperelasticMaterialBase Implementation
// ============================================================================

SymmetricTensor<4, 3> HyperelasticMaterialBase::compute_spatial_tangent(
    const Tensor<2, 3>& F) const
{
    // Push-forward of material tangent
    SymmetricTensor<4, 3> CC = compute_material_tangent(F);
    double J = determinant(F);
    
    SymmetricTensor<4, 3> c;
    c = 0;
    
    // c_ijkl = (1/J) * F_iI * F_jJ * CC_IJKL * F_kK * F_lL
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                    for (unsigned int I = 0; I < 3; ++I)
                        for (unsigned int J = 0; J < 3; ++J)
                            for (unsigned int K = 0; K < 3; ++K)
                                for (unsigned int L = 0; L < 3; ++L)
                                    c[i][j][k][l] += F[i][I] * F[j][J] * 
                                                      CC[I][J][K][L] * 
                                                      F[k][K] * F[l][L];
    
    c *= (1.0 / J);
    
    return c;
}

// ============================================================================
// NeoHookeanMaterial Implementation
// ============================================================================

NeoHookeanMaterial::NeoHookeanMaterial(const HyperelasticNeoHookeanProperties& props)
    : mu_(props.shear_modulus),
      kappa_(props.bulk_modulus),
      rho_(props.density)
{
    // λ = κ - 2μ/3
    lambda_ = kappa_ - 2.0 * mu_ / 3.0;
}

NeoHookeanMaterial::NeoHookeanMaterial(double E, double nu, double rho)
    : rho_(rho)
{
    mu_ = E / (2.0 * (1.0 + nu));
    kappa_ = E / (3.0 * (1.0 - 2.0 * nu));
    lambda_ = kappa_ - 2.0 * mu_ / 3.0;
}

NeoHookeanMaterial NeoHookeanMaterial::from_lame_parameters(
    double lambda, double mu, double rho)
{
    NeoHookeanMaterial mat(0, 0, rho);
    mat.mu_ = mu;
    mat.lambda_ = lambda;
    mat.kappa_ = lambda + 2.0 * mu / 3.0;
    return mat;
}

double NeoHookeanMaterial::compute_strain_energy(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> C = compute_right_cauchy_green(F);
    double I1 = trace(C);
    double J = determinant(F);
    double lnJ = std::log(J);
    
    // W = (μ/2)(I₁ - 3) - μ ln(J) + (λ/2)(ln J)²
    return 0.5 * mu_ * (I1 - 3.0) - mu_ * lnJ + 0.5 * lambda_ * lnJ * lnJ;
}

Tensor<2, 3> NeoHookeanMaterial::compute_pk1_stress(const Tensor<2, 3>& F) const
{
    double J = determinant(F);
    Tensor<2, 3> F_inv_T = transpose(invert(F));
    
    // P = μ*F - μ*F^{-T} + λ*ln(J)*F^{-T}
    Tensor<2, 3> P;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            P[i][j] = mu_ * F[i][j] + (lambda_ * std::log(J) - mu_) * F_inv_T[i][j];
    
    return P;
}

SymmetricTensor<2, 3> NeoHookeanMaterial::compute_pk2_stress(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> C = compute_right_cauchy_green(F);
    double J = determinant(F);
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    // Identity tensor
    SymmetricTensor<2, 3> I;
    I = 0;
    for (unsigned int i = 0; i < 3; ++i)
        I[i][i] = 1.0;
    
    // S = μ*(I - C^{-1}) + λ*ln(J)*C^{-1}
    SymmetricTensor<2, 3> S = mu_ * (I - C_inv) + lambda_ * std::log(J) * C_inv;
    
    return S;
}

SymmetricTensor<2, 3> NeoHookeanMaterial::compute_cauchy_stress(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> S = compute_pk2_stress(F);
    return push_forward(S, F);
}

SymmetricTensor<4, 3> NeoHookeanMaterial::compute_material_tangent(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> C = compute_right_cauchy_green(F);
    double J = determinant(F);
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    // CC_IJKL = λ*C^{-1}_{IJ}*C^{-1}_{KL} 
    //         + (μ - λ*ln(J))*(C^{-1}_{IK}*C^{-1}_{JL} + C^{-1}_{IL}*C^{-1}_{JK})
    
    SymmetricTensor<4, 3> CC;
    CC = 0;
    
    double factor = mu_ - lambda_ * std::log(J);
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                    CC[i][j][k][l] = lambda_ * C_inv[i][j] * C_inv[k][l] +
                        factor * (C_inv[i][k] * C_inv[j][l] + C_inv[i][l] * C_inv[j][k]);
    
    return CC;
}

// ============================================================================
// MooneyRivlinMaterial Implementation
// ============================================================================

MooneyRivlinMaterial::MooneyRivlinMaterial(const HyperelasticMooneyRivlinProperties& props)
    : C10_(props.C10),
      C01_(props.C01),
      kappa_(props.bulk_modulus),
      rho_(props.density)
{
}

MooneyRivlinMaterial::MooneyRivlinMaterial(
    double C10, double C01, double kappa, double rho)
    : C10_(C10), C01_(C01), kappa_(kappa), rho_(rho)
{
}

void MooneyRivlinMaterial::compute_deviatoric_invariants(
    const Tensor<2, 3>& F,
    double& I1_bar, double& I2_bar,
    SymmetricTensor<2, 3>& C, double& J) const
{
    C = compute_right_cauchy_green(F);
    J = determinant(F);
    
    double I1, I2, I3;
    compute_invariants(C, I1, I2, I3);
    
    double J23 = std::pow(J, -2.0/3.0);
    double J43 = J23 * J23;
    
    I1_bar = J23 * I1;
    I2_bar = J43 * I2;
}

double MooneyRivlinMaterial::compute_strain_energy(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> C;
    double J, I1_bar, I2_bar;
    compute_deviatoric_invariants(F, I1_bar, I2_bar, C, J);
    
    // W = C₁₀(Ī₁ - 3) + C₀₁(Ī₂ - 3) + (κ/2)(J - 1)²
    return C10_ * (I1_bar - 3.0) + C01_ * (I2_bar - 3.0) + 
           0.5 * kappa_ * (J - 1.0) * (J - 1.0);
}

Tensor<2, 3> MooneyRivlinMaterial::compute_pk1_stress(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> S = compute_pk2_stress(F);
    
    // P = F * S
    Tensor<2, 3> P;
    P = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                P[i][j] += F[i][k] * S[k][j];
    
    return P;
}

SymmetricTensor<2, 3> MooneyRivlinMaterial::compute_pk2_stress(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> C;
    double J, I1_bar, I2_bar;
    compute_deviatoric_invariants(F, I1_bar, I2_bar, C, J);
    
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    double J23 = std::pow(J, -2.0/3.0);
    double J43 = J23 * J23;
    
    // Identity tensor
    SymmetricTensor<2, 3> I;
    I = 0;
    for (unsigned int i = 0; i < 3; ++i)
        I[i][i] = 1.0;
    
    // Deviatoric part of S
    // S_iso = 2*C10*J^(-2/3)*(I - (1/3)*I1*C^{-1})
    //       + 2*C01*J^(-4/3)*(I1*I - C - (2/3)*I2*C^{-1})
    
    double I1 = trace(C);
    double I2 = 0.5 * (I1 * I1 - (C * C));  // Simplified I2
    
    SymmetricTensor<2, 3> S_dev;
    S_dev = 0;
    
    // First term: C10 contribution
    S_dev += 2.0 * C10_ * J23 * (I - (1.0/3.0) * I1 * C_inv);
    
    // Second term: C01 contribution
    S_dev += 2.0 * C01_ * J43 * (I1 * I - C - (2.0/3.0) * I2 * C_inv);
    
    // Volumetric part: S_vol = κ*J*(J-1)*C^{-1}
    SymmetricTensor<2, 3> S_vol = kappa_ * J * (J - 1.0) * C_inv;
    
    return S_dev + S_vol;
}

SymmetricTensor<2, 3> MooneyRivlinMaterial::compute_cauchy_stress(const Tensor<2, 3>& F) const
{
    SymmetricTensor<2, 3> S = compute_pk2_stress(F);
    return push_forward(S, F);
}

SymmetricTensor<4, 3> MooneyRivlinMaterial::compute_material_tangent(const Tensor<2, 3>& F) const
{
    // For simplicity, use numerical differentiation or approximate tangent
    // A full analytical tangent for Mooney-Rivlin is complex
    
    SymmetricTensor<2, 3> C = compute_right_cauchy_green(F);
    double J = determinant(F);
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    // Use effective modulus approach for approximate tangent
    double mu_eff = 2.0 * (C10_ + C01_);
    double lambda_eff = kappa_ - 2.0 * mu_eff / 3.0;
    
    SymmetricTensor<4, 3> CC;
    CC = 0;
    
    double factor = mu_eff - lambda_eff * std::log(J);
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                    CC[i][j][k][l] = lambda_eff * C_inv[i][j] * C_inv[k][l] +
                        factor * (C_inv[i][k] * C_inv[j][l] + C_inv[i][l] * C_inv[j][k]);
    
    // Add volumetric contribution
    double p = kappa_ * J * (J - 1.0);
    double Kp = kappa_ * (2.0 * J - 1.0);
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                    CC[i][j][k][l] += Kp * C_inv[i][j] * C_inv[k][l] +
                        2.0 * p * (C_inv[i][k] * C_inv[j][l] + C_inv[i][l] * C_inv[j][k]);
    
    return CC;
}

} // namespace FEA
