/**
 * @file linear_elastic.h
 * @brief Linear elastic material model utilities
 * 
 * Provides helper functions for isotropic linear elastic materials
 * including stress-strain calculations and tensor operations.
 */

#ifndef LINEAR_ELASTIC_H
#define LINEAR_ELASTIC_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include "../material_library.h"

namespace FEA {

using namespace dealii;

/**
 * @brief Linear elastic material point computation
 * 
 * Handles stress-strain relationships for linear elastic materials,
 * including both isotropic and small-strain cases.
 */
class LinearElasticMaterial {
public:
    /**
     * @brief Construct from isotropic properties
     */
    explicit LinearElasticMaterial(const IsotropicElasticProperties& props);
    
    /**
     * @brief Compute stress from strain
     * @param strain Small strain tensor ε
     * @return Cauchy stress tensor σ
     */
    SymmetricTensor<2, 3> compute_stress(
        const SymmetricTensor<2, 3>& strain) const;
    
    /**
     * @brief Compute stress with thermal effects
     * @param strain Mechanical strain tensor
     * @param delta_temperature Temperature change from reference
     * @return Cauchy stress tensor
     */
    SymmetricTensor<2, 3> compute_stress_with_thermal(
        const SymmetricTensor<2, 3>& strain,
        double delta_temperature) const;
    
    /**
     * @brief Get the elasticity tensor
     * @return 4th order elasticity tensor C
     */
    const SymmetricTensor<4, 3>& get_elasticity_tensor() const {
        return elasticity_tensor_;
    }
    
    /**
     * @brief Get thermal expansion tensor
     * @return 2nd order thermal expansion tensor α
     */
    SymmetricTensor<2, 3> get_thermal_expansion_tensor() const;
    
    /**
     * @brief Compute strain energy density
     * @param strain Strain tensor
     * @return Strain energy density W = 0.5 * σ : ε
     */
    double compute_strain_energy(const SymmetricTensor<2, 3>& strain) const;
    
    // Material properties access
    double get_youngs_modulus() const { return E_; }
    double get_poissons_ratio() const { return nu_; }
    double get_shear_modulus() const { return mu_; }
    double get_bulk_modulus() const { return K_; }
    double get_lambda() const { return lambda_; }
    double get_density() const { return rho_; }
    double get_thermal_expansion() const { return alpha_; }
    
private:
    double E_;       // Young's modulus
    double nu_;      // Poisson's ratio
    double mu_;      // Shear modulus
    double K_;       // Bulk modulus
    double lambda_;  // Lamé's first parameter
    double rho_;     // Density
    double alpha_;   // Thermal expansion coefficient
    
    SymmetricTensor<4, 3> elasticity_tensor_;
};

/**
 * @brief Build isotropic elasticity tensor using Lamé parameters
 * @param lambda Lamé's first parameter λ
 * @param mu Shear modulus μ
 * @return 4th order isotropic elasticity tensor
 */
SymmetricTensor<4, 3> build_isotropic_elasticity_tensor(double lambda, double mu);

/**
 * @brief Build compliance tensor from elasticity tensor
 * @param C Elasticity tensor
 * @return Compliance tensor S such that S:C = I (4th order identity)
 */
SymmetricTensor<4, 3> invert_elasticity_tensor(const SymmetricTensor<4, 3>& C);

/**
 * @brief Compute plane stress elasticity tensor
 * 
 * For thin plates/shells where σ_33 = 0
 * 
 * @param E Young's modulus
 * @param nu Poisson's ratio
 * @return 2D plane stress elasticity tensor (3x3 for [σ11, σ22, σ12])
 */
SymmetricTensor<4, 2> build_plane_stress_tensor(double E, double nu);

/**
 * @brief Compute plane strain elasticity tensor
 * 
 * For long prismatic bodies where ε_33 = 0
 * 
 * @param E Young's modulus
 * @param nu Poisson's ratio
 * @return 2D plane strain elasticity tensor
 */
SymmetricTensor<4, 2> build_plane_strain_tensor(double E, double nu);

/**
 * @brief Convert engineering constants to Lamé parameters
 */
inline std::pair<double, double> engineering_to_lame(double E, double nu) {
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));
    return {lambda, mu};
}

/**
 * @brief Convert Lamé parameters to engineering constants
 */
inline std::pair<double, double> lame_to_engineering(double lambda, double mu) {
    double E = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
    double nu = lambda / (2.0 * (lambda + mu));
    return {E, nu};
}

} // namespace FEA

#endif // LINEAR_ELASTIC_H
