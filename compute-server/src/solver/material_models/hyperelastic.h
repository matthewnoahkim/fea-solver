/**
 * @file hyperelastic.h
 * @brief Hyperelastic material models for large deformation analysis
 * 
 * Implements:
 * - Neo-Hookean (compressible)
 * - Mooney-Rivlin (2-parameter)
 * 
 * These models are formulated in terms of strain energy density functions
 * and work with the deformation gradient F for geometric nonlinearity.
 */

#ifndef HYPERELASTIC_H
#define HYPERELASTIC_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include "../material_library.h"

namespace FEA {

using namespace dealii;

/**
 * @brief Base class for hyperelastic material models
 * 
 * Provides interface for strain energy density functions
 * and stress/tangent computations.
 */
class HyperelasticMaterialBase {
public:
    virtual ~HyperelasticMaterialBase() = default;
    
    /**
     * @brief Compute strain energy density W(F)
     * @param F Deformation gradient
     * @return Strain energy density [J/m³]
     */
    virtual double compute_strain_energy(const Tensor<2, 3>& F) const = 0;
    
    /**
     * @brief Compute 1st Piola-Kirchhoff stress P
     * 
     * P = ∂W/∂F
     * 
     * @param F Deformation gradient
     * @return 1st Piola-Kirchhoff stress tensor (not symmetric)
     */
    virtual Tensor<2, 3> compute_pk1_stress(const Tensor<2, 3>& F) const = 0;
    
    /**
     * @brief Compute 2nd Piola-Kirchhoff stress S
     * 
     * S = F^{-1} * P = 2 * ∂W/∂C
     * 
     * @param F Deformation gradient
     * @return 2nd Piola-Kirchhoff stress tensor (symmetric)
     */
    virtual SymmetricTensor<2, 3> compute_pk2_stress(const Tensor<2, 3>& F) const = 0;
    
    /**
     * @brief Compute Cauchy (true) stress σ
     * 
     * σ = (1/J) * F * S * F^T = (1/J) * P * F^T
     * 
     * @param F Deformation gradient
     * @return Cauchy stress tensor
     */
    virtual SymmetricTensor<2, 3> compute_cauchy_stress(const Tensor<2, 3>& F) const = 0;
    
    /**
     * @brief Compute material tangent modulus
     * 
     * CC = 2 * ∂S/∂C = 4 * ∂²W/∂C∂C
     * 
     * @param F Deformation gradient
     * @return Material tangent tensor
     */
    virtual SymmetricTensor<4, 3> compute_material_tangent(const Tensor<2, 3>& F) const = 0;
    
    /**
     * @brief Compute spatial tangent modulus
     * 
     * c = (1/J) * F⊗F : CC : F^T⊗F^T (push-forward of material tangent)
     * 
     * @param F Deformation gradient
     * @return Spatial tangent tensor
     */
    virtual SymmetricTensor<4, 3> compute_spatial_tangent(const Tensor<2, 3>& F) const;
    
    /**
     * @brief Get initial shear modulus
     */
    virtual double get_initial_shear_modulus() const = 0;
    
    /**
     * @brief Get bulk modulus
     */
    virtual double get_bulk_modulus() const = 0;
    
    /**
     * @brief Get density
     */
    virtual double get_density() const = 0;
};

/**
 * @brief Neo-Hookean hyperelastic material
 * 
 * Strain energy density:
 * W = (μ/2)(I₁ - 3) - μ ln(J) + (λ/2)(ln J)²
 * 
 * where:
 * - I₁ = tr(C) = tr(F^T F), first invariant of right Cauchy-Green
 * - J = det(F), volume ratio
 * - μ, λ are Lamé parameters
 */
class NeoHookeanMaterial : public HyperelasticMaterialBase {
public:
    /**
     * @brief Construct from properties struct
     */
    explicit NeoHookeanMaterial(const HyperelasticNeoHookeanProperties& props);
    
    /**
     * @brief Construct from engineering constants
     */
    NeoHookeanMaterial(double E, double nu, double rho);
    
    /**
     * @brief Construct from Lamé parameters
     */
    static NeoHookeanMaterial from_lame_parameters(double lambda, double mu, double rho);
    
    // Implement base class interface
    double compute_strain_energy(const Tensor<2, 3>& F) const override;
    Tensor<2, 3> compute_pk1_stress(const Tensor<2, 3>& F) const override;
    SymmetricTensor<2, 3> compute_pk2_stress(const Tensor<2, 3>& F) const override;
    SymmetricTensor<2, 3> compute_cauchy_stress(const Tensor<2, 3>& F) const override;
    SymmetricTensor<4, 3> compute_material_tangent(const Tensor<2, 3>& F) const override;
    
    double get_initial_shear_modulus() const override { return mu_; }
    double get_bulk_modulus() const override { return kappa_; }
    double get_density() const override { return rho_; }
    
    // Additional accessors
    double get_lambda() const { return lambda_; }
    double get_mu() const { return mu_; }
    
private:
    double mu_;      // Shear modulus
    double kappa_;   // Bulk modulus
    double lambda_;  // Lamé's first parameter
    double rho_;     // Density
};

/**
 * @brief Mooney-Rivlin hyperelastic material
 * 
 * Strain energy density (compressible formulation):
 * W = C₁₀(Ī₁ - 3) + C₀₁(Ī₂ - 3) + (κ/2)(J - 1)²
 * 
 * where:
 * - Ī₁ = J^(-2/3) * I₁, first deviatoric invariant
 * - Ī₂ = J^(-4/3) * I₂, second deviatoric invariant
 * - I₁ = tr(C), I₂ = 0.5*(tr(C)² - tr(C²))
 * - C₁₀, C₀₁ are material constants
 * - κ is bulk modulus
 * 
 * Initial shear modulus: μ = 2(C₁₀ + C₀₁)
 */
class MooneyRivlinMaterial : public HyperelasticMaterialBase {
public:
    /**
     * @brief Construct from properties struct
     */
    explicit MooneyRivlinMaterial(const HyperelasticMooneyRivlinProperties& props);
    
    /**
     * @brief Construct from material constants
     */
    MooneyRivlinMaterial(double C10, double C01, double kappa, double rho);
    
    // Implement base class interface
    double compute_strain_energy(const Tensor<2, 3>& F) const override;
    Tensor<2, 3> compute_pk1_stress(const Tensor<2, 3>& F) const override;
    SymmetricTensor<2, 3> compute_pk2_stress(const Tensor<2, 3>& F) const override;
    SymmetricTensor<2, 3> compute_cauchy_stress(const Tensor<2, 3>& F) const override;
    SymmetricTensor<4, 3> compute_material_tangent(const Tensor<2, 3>& F) const override;
    
    double get_initial_shear_modulus() const override { return 2.0 * (C10_ + C01_); }
    double get_bulk_modulus() const override { return kappa_; }
    double get_density() const override { return rho_; }
    
    // Additional accessors
    double get_C10() const { return C10_; }
    double get_C01() const { return C01_; }
    
private:
    double C10_;     // First Mooney-Rivlin constant
    double C01_;     // Second Mooney-Rivlin constant
    double kappa_;   // Bulk modulus
    double rho_;     // Density
    
    /**
     * @brief Compute deviatoric invariants
     */
    void compute_deviatoric_invariants(
        const Tensor<2, 3>& F,
        double& I1_bar, double& I2_bar,
        SymmetricTensor<2, 3>& C, double& J) const;
};

// ============================================================================
// Utility Functions for Large Deformation Analysis
// ============================================================================

/**
 * @brief Compute right Cauchy-Green deformation tensor
 * 
 * C = F^T * F
 */
inline SymmetricTensor<2, 3> compute_right_cauchy_green(const Tensor<2, 3>& F) {
    SymmetricTensor<2, 3> C;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            C[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                C[i][j] += F[k][i] * F[k][j];
        }
    return C;
}

/**
 * @brief Compute left Cauchy-Green deformation tensor
 * 
 * B = F * F^T
 */
inline SymmetricTensor<2, 3> compute_left_cauchy_green(const Tensor<2, 3>& F) {
    SymmetricTensor<2, 3> B;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            B[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                B[i][j] += F[i][k] * F[j][k];
        }
    return B;
}

/**
 * @brief Compute Green-Lagrange strain tensor
 * 
 * E = 0.5 * (C - I) = 0.5 * (F^T F - I)
 */
inline SymmetricTensor<2, 3> compute_green_lagrange_strain(const Tensor<2, 3>& F) {
    SymmetricTensor<2, 3> C = compute_right_cauchy_green(F);
    SymmetricTensor<2, 3> E;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            E[i][j] = 0.5 * C[i][j];
            if (i == j) E[i][j] -= 0.5;
        }
    return E;
}

/**
 * @brief Compute Almansi (Euler) strain tensor
 * 
 * e = 0.5 * (I - B^{-1})
 */
SymmetricTensor<2, 3> compute_almansi_strain(const Tensor<2, 3>& F);

/**
 * @brief Push forward a 2nd order tensor from reference to current config
 * 
 * a = (1/J) * F * A * F^T
 */
SymmetricTensor<2, 3> push_forward(
    const SymmetricTensor<2, 3>& A,
    const Tensor<2, 3>& F);

/**
 * @brief Pull back a 2nd order tensor from current to reference config
 * 
 * A = J * F^{-1} * a * F^{-T}
 */
SymmetricTensor<2, 3> pull_back(
    const SymmetricTensor<2, 3>& a,
    const Tensor<2, 3>& F);

/**
 * @brief Compute invariants of a symmetric tensor
 * 
 * @param A Symmetric tensor
 * @param I1 First invariant: tr(A)
 * @param I2 Second invariant: 0.5*(tr(A)² - tr(A²))
 * @param I3 Third invariant: det(A)
 */
void compute_invariants(
    const SymmetricTensor<2, 3>& A,
    double& I1, double& I2, double& I3);

/**
 * @brief Compute principal stretches from deformation gradient
 * 
 * λ₁, λ₂, λ₃ are square roots of eigenvalues of C
 */
std::array<double, 3> compute_principal_stretches(const Tensor<2, 3>& F);

} // namespace FEA

#endif // HYPERELASTIC_H
