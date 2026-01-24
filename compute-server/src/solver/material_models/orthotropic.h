/**
 * @file orthotropic.h
 * @brief Orthotropic linear elastic material model
 * 
 * Supports materials with three mutually perpendicular planes of symmetry:
 * - Unidirectional fiber composites (CFRP, GFRP)
 * - Wood along grain/radial/tangential directions
 * - Rolled sheet metals
 */

#ifndef ORTHOTROPIC_H
#define ORTHOTROPIC_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include "../material_library.h"
#include <array>

namespace FEA {

using namespace dealii;

/**
 * @brief Orthotropic material computation at a material point
 * 
 * Handles stress-strain relationships for orthotropic materials
 * with proper orientation transformations.
 */
class OrthotropicMaterial {
public:
    /**
     * @brief Construct from orthotropic properties
     */
    explicit OrthotropicMaterial(const OrthotropicElasticProperties& props);
    
    /**
     * @brief Compute stress from strain (in global coordinates)
     * @param strain Small strain tensor ε (global)
     * @return Cauchy stress tensor σ (global)
     */
    SymmetricTensor<2, 3> compute_stress(
        const SymmetricTensor<2, 3>& strain) const;
    
    /**
     * @brief Compute stress with thermal effects
     * @param strain Total strain tensor (global)
     * @param delta_temperature Temperature change from reference
     * @return Cauchy stress tensor
     */
    SymmetricTensor<2, 3> compute_stress_with_thermal(
        const SymmetricTensor<2, 3>& strain,
        double delta_temperature) const;
    
    /**
     * @brief Get elasticity tensor in global coordinates
     */
    const SymmetricTensor<4, 3>& get_elasticity_tensor() const {
        return C_global_;
    }
    
    /**
     * @brief Get elasticity tensor in material coordinates
     */
    const SymmetricTensor<4, 3>& get_material_elasticity_tensor() const {
        return C_material_;
    }
    
    /**
     * @brief Get thermal expansion tensor (global coordinates)
     */
    const SymmetricTensor<2, 3>& get_thermal_expansion_tensor() const {
        return alpha_global_;
    }
    
    /**
     * @brief Rotate strain from global to material coordinates
     */
    SymmetricTensor<2, 3> rotate_to_material(
        const SymmetricTensor<2, 3>& tensor_global) const;
    
    /**
     * @brief Rotate stress from material to global coordinates
     */
    SymmetricTensor<2, 3> rotate_to_global(
        const SymmetricTensor<2, 3>& tensor_material) const;
    
    /**
     * @brief Compute strain energy density
     * @param strain Strain tensor (global coordinates)
     * @return Strain energy density W
     */
    double compute_strain_energy(const SymmetricTensor<2, 3>& strain) const;
    
    /**
     * @brief Get the orientation rotation matrix
     */
    const Tensor<2, 3>& get_orientation_matrix() const {
        return R_;
    }
    
    /**
     * @brief Update orientation (e.g., for fiber-aligned analysis)
     */
    void set_orientation(const Tensor<2, 3>& rotation_matrix);
    
    /**
     * @brief Check if properties are thermodynamically valid
     */
    bool is_valid() const;
    
    // Property access (in material coordinates)
    double get_E1() const { return E1_; }
    double get_E2() const { return E2_; }
    double get_E3() const { return E3_; }
    double get_G12() const { return G12_; }
    double get_G13() const { return G13_; }
    double get_G23() const { return G23_; }
    double get_nu12() const { return nu12_; }
    double get_nu13() const { return nu13_; }
    double get_nu23() const { return nu23_; }
    double get_density() const { return rho_; }
    
private:
    // Material properties in principal directions
    double E1_, E2_, E3_;
    double nu12_, nu13_, nu23_;
    double G12_, G13_, G23_;
    double rho_;
    double alpha1_, alpha2_, alpha3_;
    
    // Rotation matrix from global to material
    Tensor<2, 3> R_;
    
    // Stiffness tensors
    SymmetricTensor<4, 3> C_material_;  // In material coordinates
    SymmetricTensor<4, 3> C_global_;    // In global coordinates
    
    // Thermal expansion (global)
    SymmetricTensor<2, 3> alpha_global_;
    
    // Build stiffness matrix in material coordinates
    void build_material_stiffness();
    
    // Rotate stiffness to global coordinates
    void rotate_stiffness_to_global();
    
    // Rotate thermal expansion to global
    void rotate_thermal_expansion_to_global();
};

/**
 * @brief Build 6x6 Voigt stiffness matrix for orthotropic material
 * 
 * @param E1, E2, E3 Young's moduli in principal directions
 * @param nu12, nu13, nu23 Poisson's ratios
 * @param G12, G13, G23 Shear moduli
 * @return 6x6 stiffness matrix in Voigt notation
 */
FullMatrix<double> build_orthotropic_voigt_matrix(
    double E1, double E2, double E3,
    double nu12, double nu13, double nu23,
    double G12, double G13, double G23);

/**
 * @brief Convert 6x6 Voigt matrix to 4th order tensor
 */
SymmetricTensor<4, 3> voigt_to_tensor(const FullMatrix<double>& voigt);

/**
 * @brief Convert 4th order tensor to 6x6 Voigt matrix
 */
FullMatrix<double> tensor_to_voigt(const SymmetricTensor<4, 3>& tensor);

/**
 * @brief Build Bond transformation matrix for rotating stiffness
 * 
 * For rotating C' = T * C * T^T where T is the 6x6 Bond matrix
 * 
 * @param R 3x3 rotation matrix (transforms vectors from old to new coords)
 * @return 6x6 Bond transformation matrix for stiffness
 */
FullMatrix<double> build_bond_transformation_matrix(const Tensor<2, 3>& R);

/**
 * @brief Check orthotropic positive definiteness conditions
 * 
 * Verifies:
 * - All moduli positive
 * - Poisson ratio bounds
 * - Determinant condition
 */
bool check_orthotropic_constraints(
    double E1, double E2, double E3,
    double nu12, double nu13, double nu23,
    double G12, double G13, double G23);

/**
 * @brief Compute transversely isotropic properties
 * 
 * Special case where E2 = E3, G12 = G13, nu12 = nu13
 * (Common for unidirectional composites)
 * 
 * @param E_L Longitudinal (fiber) modulus
 * @param E_T Transverse modulus
 * @param nu_LT Major Poisson's ratio
 * @param G_LT Longitudinal shear modulus
 * @param nu_TT Transverse Poisson's ratio (G_TT = E_T / (2*(1+nu_TT)))
 */
OrthotropicElasticProperties create_transversely_isotropic(
    double E_L, double E_T, double nu_LT, double G_LT, double nu_TT);

} // namespace FEA

#endif // ORTHOTROPIC_H
