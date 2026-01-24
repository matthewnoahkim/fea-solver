/**
 * @file orthotropic.cc
 * @brief Implementation of orthotropic material model
 */

#include "orthotropic.h"
#include <cmath>
#include <stdexcept>

namespace FEA {

// ============================================================================
// OrthotropicMaterial Implementation
// ============================================================================

OrthotropicMaterial::OrthotropicMaterial(const OrthotropicElasticProperties& props)
    : E1_(props.E1), E2_(props.E2), E3_(props.E3),
      nu12_(props.nu12), nu13_(props.nu13), nu23_(props.nu23),
      G12_(props.G12), G13_(props.G13), G23_(props.G23),
      rho_(props.density),
      alpha1_(props.alpha1), alpha2_(props.alpha2), alpha3_(props.alpha3),
      R_(props.orientation_matrix)
{
    build_material_stiffness();
    rotate_stiffness_to_global();
    rotate_thermal_expansion_to_global();
}

void OrthotropicMaterial::build_material_stiffness()
{
    FullMatrix<double> C_voigt = build_orthotropic_voigt_matrix(
        E1_, E2_, E3_, nu12_, nu13_, nu23_, G12_, G13_, G23_);
    
    C_material_ = voigt_to_tensor(C_voigt);
}

void OrthotropicMaterial::rotate_stiffness_to_global()
{
    // C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs
    C_global_ = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                    for (unsigned int p = 0; p < 3; ++p)
                        for (unsigned int q = 0; q < 3; ++q)
                            for (unsigned int r = 0; r < 3; ++r)
                                for (unsigned int s = 0; s < 3; ++s)
                                    C_global_[i][j][k][l] += 
                                        R_[i][p] * R_[j][q] * R_[k][r] * R_[l][s] * 
                                        C_material_[p][q][r][s];
}

void OrthotropicMaterial::rotate_thermal_expansion_to_global()
{
    // Thermal expansion in material coordinates
    SymmetricTensor<2, 3> alpha_mat;
    alpha_mat = 0;
    alpha_mat[0][0] = alpha1_;
    alpha_mat[1][1] = alpha2_;
    alpha_mat[2][2] = alpha3_;
    
    // Rotate: α'_ij = R_ip R_jq α_pq
    alpha_global_ = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    alpha_global_[i][j] += R_[i][p] * R_[j][q] * alpha_mat[p][q];
}

SymmetricTensor<2, 3> OrthotropicMaterial::compute_stress(
    const SymmetricTensor<2, 3>& strain) const
{
    return C_global_ * strain;
}

SymmetricTensor<2, 3> OrthotropicMaterial::compute_stress_with_thermal(
    const SymmetricTensor<2, 3>& strain,
    double delta_temperature) const
{
    // Thermal strain
    SymmetricTensor<2, 3> thermal_strain = alpha_global_ * delta_temperature;
    
    // Mechanical strain
    SymmetricTensor<2, 3> mech_strain = strain - thermal_strain;
    
    return C_global_ * mech_strain;
}

SymmetricTensor<2, 3> OrthotropicMaterial::rotate_to_material(
    const SymmetricTensor<2, 3>& tensor_global) const
{
    // ε_mat = R^T ε_global R
    SymmetricTensor<2, 3> tensor_material;
    tensor_material = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    tensor_material[i][j] += R_[p][i] * R_[q][j] * tensor_global[p][q];
    
    return tensor_material;
}

SymmetricTensor<2, 3> OrthotropicMaterial::rotate_to_global(
    const SymmetricTensor<2, 3>& tensor_material) const
{
    // σ_global = R σ_mat R^T
    SymmetricTensor<2, 3> tensor_global;
    tensor_global = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    tensor_global[i][j] += R_[i][p] * R_[j][q] * tensor_material[p][q];
    
    return tensor_global;
}

double OrthotropicMaterial::compute_strain_energy(
    const SymmetricTensor<2, 3>& strain) const
{
    SymmetricTensor<2, 3> stress = compute_stress(strain);
    return 0.5 * (stress * strain);
}

void OrthotropicMaterial::set_orientation(const Tensor<2, 3>& rotation_matrix)
{
    R_ = rotation_matrix;
    rotate_stiffness_to_global();
    rotate_thermal_expansion_to_global();
}

bool OrthotropicMaterial::is_valid() const
{
    return check_orthotropic_constraints(
        E1_, E2_, E3_, nu12_, nu13_, nu23_, G12_, G13_, G23_);
}

// ============================================================================
// Free Functions
// ============================================================================

FullMatrix<double> build_orthotropic_voigt_matrix(
    double E1, double E2, double E3,
    double nu12, double nu13, double nu23,
    double G12, double G13, double G23)
{
    // Compute derived Poisson's ratios from symmetry
    double nu21 = nu12 * E2 / E1;
    double nu31 = nu13 * E3 / E1;
    double nu32 = nu23 * E3 / E2;
    
    // Determinant for compliance inversion
    double Delta = 1.0 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2.0*nu12*nu23*nu31;
    
    if (std::abs(Delta) < 1e-15) {
        throw std::runtime_error("Orthotropic material is ill-conditioned");
    }
    
    // Stiffness components
    double C11 = E1 * (1.0 - nu23*nu32) / Delta;
    double C22 = E2 * (1.0 - nu13*nu31) / Delta;
    double C33 = E3 * (1.0 - nu12*nu21) / Delta;
    double C12 = E1 * (nu21 + nu31*nu23) / Delta;
    double C13 = E1 * (nu31 + nu21*nu32) / Delta;
    double C23 = E2 * (nu32 + nu12*nu31) / Delta;
    double C44 = G23;
    double C55 = G13;
    double C66 = G12;
    
    // Build 6x6 Voigt matrix
    // Order: [σ11, σ22, σ33, σ23, σ13, σ12]
    FullMatrix<double> C(6, 6);
    C = 0;
    
    C(0, 0) = C11;  C(0, 1) = C12;  C(0, 2) = C13;
    C(1, 0) = C12;  C(1, 1) = C22;  C(1, 2) = C23;
    C(2, 0) = C13;  C(2, 1) = C23;  C(2, 2) = C33;
    C(3, 3) = C44;
    C(4, 4) = C55;
    C(5, 5) = C66;
    
    return C;
}

SymmetricTensor<4, 3> voigt_to_tensor(const FullMatrix<double>& voigt)
{
    // Voigt indices: 0=11, 1=22, 2=33, 3=23, 4=13, 5=12
    auto voigt_index = [](unsigned int i, unsigned int j) -> unsigned int {
        if (i == j) return i;
        if ((i == 1 && j == 2) || (i == 2 && j == 1)) return 3;
        if ((i == 0 && j == 2) || (i == 2 && j == 0)) return 4;
        if ((i == 0 && j == 1) || (i == 1 && j == 0)) return 5;
        return 0;
    };
    
    SymmetricTensor<4, 3> tensor;
    tensor = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l) {
                    unsigned int I = voigt_index(i, j);
                    unsigned int J = voigt_index(k, l);
                    
                    // Factors for shear components
                    double factor = 1.0;
                    if (I > 2 && J > 2) factor = 1.0;  // Both shear
                    else if (I > 2 || J > 2) factor = 1.0;  // One shear
                    
                    // Additional factor for tensor symmetry
                    if (i != j && k != l) factor *= 0.25;
                    else if (i != j || k != l) factor *= 0.5;
                    
                    tensor[i][j][k][l] = voigt(I, J) * factor;
                }
    
    return tensor;
}

FullMatrix<double> tensor_to_voigt(const SymmetricTensor<4, 3>& tensor)
{
    FullMatrix<double> voigt(6, 6);
    voigt = 0;
    
    // Index mapping
    const unsigned int i_map[6] = {0, 1, 2, 1, 0, 0};
    const unsigned int j_map[6] = {0, 1, 2, 2, 2, 1};
    
    for (unsigned int I = 0; I < 6; ++I)
        for (unsigned int J = 0; J < 6; ++J) {
            unsigned int i = i_map[I];
            unsigned int j = j_map[I];
            unsigned int k = i_map[J];
            unsigned int l = j_map[J];
            
            double factor = 1.0;
            if (I > 2) factor *= 2.0;
            if (J > 2) factor *= 2.0;
            
            voigt(I, J) = tensor[i][j][k][l] * factor;
        }
    
    return voigt;
}

FullMatrix<double> build_bond_transformation_matrix(const Tensor<2, 3>& R)
{
    // Bond transformation matrix for stiffness
    // C' = T * C * T^T
    
    FullMatrix<double> T(6, 6);
    T = 0;
    
    // Extract rotation matrix components
    double m[3][3];
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            m[i][j] = R[i][j];
    
    // Normal stress rows
    for (unsigned int i = 0; i < 3; ++i) {
        T(i, 0) = m[i][0] * m[i][0];
        T(i, 1) = m[i][1] * m[i][1];
        T(i, 2) = m[i][2] * m[i][2];
        T(i, 3) = 2.0 * m[i][1] * m[i][2];
        T(i, 4) = 2.0 * m[i][0] * m[i][2];
        T(i, 5) = 2.0 * m[i][0] * m[i][1];
    }
    
    // Shear stress rows (using complementary indices)
    const unsigned int p[3] = {1, 0, 0};
    const unsigned int q[3] = {2, 2, 1};
    
    for (unsigned int i = 0; i < 3; ++i) {
        unsigned int a = p[i];
        unsigned int b = q[i];
        T(3+i, 0) = m[a][0] * m[b][0];
        T(3+i, 1) = m[a][1] * m[b][1];
        T(3+i, 2) = m[a][2] * m[b][2];
        T(3+i, 3) = m[a][1] * m[b][2] + m[a][2] * m[b][1];
        T(3+i, 4) = m[a][0] * m[b][2] + m[a][2] * m[b][0];
        T(3+i, 5) = m[a][0] * m[b][1] + m[a][1] * m[b][0];
    }
    
    return T;
}

bool check_orthotropic_constraints(
    double E1, double E2, double E3,
    double nu12, double nu13, double nu23,
    double G12, double G13, double G23)
{
    // All moduli must be positive
    if (E1 <= 0 || E2 <= 0 || E3 <= 0) return false;
    if (G12 <= 0 || G13 <= 0 || G23 <= 0) return false;
    
    // Poisson ratio bounds: |nu_ij| < sqrt(E_i / E_j)
    if (std::abs(nu12) >= std::sqrt(E1 / E2)) return false;
    if (std::abs(nu13) >= std::sqrt(E1 / E3)) return false;
    if (std::abs(nu23) >= std::sqrt(E2 / E3)) return false;
    
    // Derived ratios
    double nu21 = nu12 * E2 / E1;
    double nu31 = nu13 * E3 / E1;
    double nu32 = nu23 * E3 / E2;
    
    // Positive definiteness: det > 0
    double det = 1.0 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2.0*nu21*nu32*nu13;
    if (det <= 0) return false;
    
    return true;
}

OrthotropicElasticProperties create_transversely_isotropic(
    double E_L, double E_T, double nu_LT, double G_LT, double nu_TT)
{
    OrthotropicElasticProperties props;
    
    props.E1 = E_L;
    props.E2 = E_T;
    props.E3 = E_T;
    
    props.nu12 = nu_LT;
    props.nu13 = nu_LT;
    props.nu23 = nu_TT;
    
    props.G12 = G_LT;
    props.G13 = G_LT;
    props.G23 = E_T / (2.0 * (1.0 + nu_TT));  // From isotropy in transverse plane
    
    return props;
}

} // namespace FEA
