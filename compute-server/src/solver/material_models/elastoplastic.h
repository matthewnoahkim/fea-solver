/**
 * @file elastoplastic.h
 * @brief J2 (von Mises) elastoplasticity material model
 * 
 * Implements isotropic elasticity with von Mises (J2) yield criterion
 * and various hardening laws for ductile metals.
 * 
 * Features:
 * - Perfect plasticity (no hardening)
 * - Linear isotropic hardening
 * - Power-law (Swift/Hollomon) hardening
 * - Tabular hardening from test data
 * - Kinematic hardening (Prager model)
 * - Combined isotropic/kinematic hardening
 */

#ifndef ELASTOPLASTIC_H
#define ELASTOPLASTIC_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include "../material_library.h"
#include <array>

namespace FEA {

using namespace dealii;

/**
 * @brief Internal state variables for plasticity
 * 
 * Stores the history-dependent state at each quadrature point.
 */
struct PlasticityState {
    SymmetricTensor<2, 3> plastic_strain;      // Accumulated plastic strain tensor
    SymmetricTensor<2, 3> back_stress;         // Kinematic hardening back stress α
    double equiv_plastic_strain;                // Equivalent plastic strain ε_p^eq
    bool yielded;                               // Whether point has yielded
    
    PlasticityState()
        : equiv_plastic_strain(0), yielded(false) {
        plastic_strain = 0;
        back_stress = 0;
    }
};

/**
 * @brief Elastoplastic material point computation
 * 
 * Handles stress update and consistent tangent for J2 plasticity
 * using a radial return mapping algorithm.
 */
class ElastoplasticMaterial {
public:
    /**
     * @brief Construct from elastoplastic properties
     */
    explicit ElastoplasticMaterial(const ElastoplasticVonMisesProperties& props);
    
    /**
     * @brief Perform stress update (radial return algorithm)
     * 
     * Given total strain, update stress and internal state variables.
     * This modifies the state in place.
     * 
     * @param total_strain Current total strain tensor
     * @param state Internal state variables (updated in place)
     * @param stress Output: computed stress tensor
     * @return true if plastic, false if elastic step
     */
    bool compute_stress_update(
        const SymmetricTensor<2, 3>& total_strain,
        PlasticityState& state,
        SymmetricTensor<2, 3>& stress) const;
    
    /**
     * @brief Compute stress with thermal effects
     * 
     * @param total_strain Total strain tensor
     * @param delta_temperature Temperature change
     * @param state Internal state (updated)
     * @param stress Output stress
     * @return true if plastic
     */
    bool compute_stress_with_thermal(
        const SymmetricTensor<2, 3>& total_strain,
        double delta_temperature,
        PlasticityState& state,
        SymmetricTensor<2, 3>& stress) const;
    
    /**
     * @brief Get algorithmic tangent modulus
     * 
     * Returns the consistent (algorithmic) tangent ∂σ/∂ε
     * for use in Newton-Raphson iterations.
     * 
     * @param strain Current strain
     * @param state Current internal state
     * @return Consistent tangent tensor
     */
    SymmetricTensor<4, 3> get_consistent_tangent(
        const SymmetricTensor<2, 3>& strain,
        const PlasticityState& state) const;
    
    /**
     * @brief Get elastic tangent (for elastic steps or initial stiffness)
     */
    const SymmetricTensor<4, 3>& get_elastic_tangent() const {
        return C_elastic_;
    }
    
    /**
     * @brief Compute von Mises equivalent stress
     */
    static double compute_von_mises(const SymmetricTensor<2, 3>& stress);
    
    /**
     * @brief Compute von Mises equivalent stress relative to back stress
     */
    double compute_shifted_von_mises(
        const SymmetricTensor<2, 3>& stress,
        const SymmetricTensor<2, 3>& back_stress) const;
    
    /**
     * @brief Get current yield stress given plastic strain
     */
    double get_yield_stress(double equiv_plastic_strain) const;
    
    /**
     * @brief Get hardening modulus dσ_y/dε_p
     */
    double get_hardening_modulus(double equiv_plastic_strain) const;
    
    // Property access
    double get_youngs_modulus() const { return E_; }
    double get_poissons_ratio() const { return nu_; }
    double get_initial_yield_stress() const { return sigma_y0_; }
    double get_isotropic_hardening() const { return H_; }
    double get_kinematic_hardening() const { return C_kin_; }
    
private:
    // Elastic properties
    double E_;      // Young's modulus
    double nu_;     // Poisson's ratio
    double mu_;     // Shear modulus
    double K_;      // Bulk modulus
    double lambda_; // Lamé parameter
    
    // Plastic properties
    double sigma_y0_;   // Initial yield stress
    double H_;          // Isotropic hardening modulus
    double C_kin_;      // Kinematic hardening modulus
    
    // Hardening law parameters
    ElastoplasticVonMisesProperties::HardeningType hardening_type_;
    double power_n_;    // Power law exponent
    double eps_0_;      // Reference strain for power law
    std::vector<std::pair<double, double>> hardening_curve_;
    
    // Elastic stiffness tensor
    SymmetricTensor<4, 3> C_elastic_;
    
    // Thermal expansion
    double alpha_;
    
    /**
     * @brief Radial return algorithm
     * 
     * Given trial stress, perform return mapping to yield surface
     * 
     * @param trial_stress Trial (elastic) stress
     * @param state Internal state (updated)
     * @param stress_n Previous stress
     * @return Final stress after return
     */
    SymmetricTensor<2, 3> radial_return(
        const SymmetricTensor<2, 3>& trial_stress,
        PlasticityState& state,
        const SymmetricTensor<2, 3>& stress_n) const;
    
    /**
     * @brief Solve for plastic multiplier increment
     */
    double solve_plastic_multiplier(
        double trial_mises,
        double yield_stress,
        double equiv_plastic_strain) const;
};

/**
 * @brief Compute deviatoric part of symmetric tensor
 */
inline SymmetricTensor<2, 3> deviator(const SymmetricTensor<2, 3>& tensor) {
    SymmetricTensor<2, 3> dev = tensor;
    double tr = trace(tensor) / 3.0;
    for (unsigned int i = 0; i < 3; ++i) {
        dev[i][i] -= tr;
    }
    return dev;
}

/**
 * @brief Compute L2 norm of symmetric tensor
 */
inline double tensor_norm(const SymmetricTensor<2, 3>& tensor) {
    return std::sqrt(tensor * tensor);
}

/**
 * @brief 4th order identity tensor for symmetric tensors
 * 
 * I_ijkl = 0.5 * (δ_ik*δ_jl + δ_il*δ_jk)
 */
SymmetricTensor<4, 3> symmetric_identity_4();

/**
 * @brief 4th order deviatoric projection tensor
 * 
 * P_ijkl = I_ijkl - (1/3)*δ_ij*δ_kl
 */
SymmetricTensor<4, 3> deviatoric_projector();

/**
 * @brief Outer product of two 2nd order symmetric tensors
 * 
 * (A ⊗ B)_ijkl = A_ij * B_kl
 */
SymmetricTensor<4, 3> outer_product(
    const SymmetricTensor<2, 3>& A,
    const SymmetricTensor<2, 3>& B);

} // namespace FEA

#endif // ELASTOPLASTIC_H
