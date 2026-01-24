/**
 * @file material_library.cc
 * @brief Implementation of material library and constitutive models
 */

#include "material_library.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace FEA {

// ============================================================================
// String Conversions
// ============================================================================

std::string material_model_to_string(MaterialModel model) {
    switch (model) {
        case MaterialModel::LINEAR_ELASTIC_ISOTROPIC: 
            return "linear_elastic_isotropic";
        case MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC: 
            return "linear_elastic_orthotropic";
        case MaterialModel::ELASTOPLASTIC_VONMISES: 
            return "elastoplastic_vonmises";
        case MaterialModel::ELASTOPLASTIC_DRUCKER_PRAGER: 
            return "elastoplastic_drucker_prager";
        case MaterialModel::HYPERELASTIC_NEOHOOKEAN: 
            return "hyperelastic_neohookean";
        case MaterialModel::HYPERELASTIC_MOONEY_RIVLIN: 
            return "hyperelastic_mooney_rivlin";
        default: 
            return "unknown";
    }
}

MaterialModel string_to_material_model(const std::string& str) {
    if (str == "linear_elastic_isotropic" || str == "linear_elastic") 
        return MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
    if (str == "linear_elastic_orthotropic" || str == "orthotropic") 
        return MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC;
    if (str == "elastoplastic_vonmises" || str == "elastoplastic") 
        return MaterialModel::ELASTOPLASTIC_VONMISES;
    if (str == "elastoplastic_drucker_prager") 
        return MaterialModel::ELASTOPLASTIC_DRUCKER_PRAGER;
    if (str == "hyperelastic_neohookean" || str == "neo_hookean") 
        return MaterialModel::HYPERELASTIC_NEOHOOKEAN;
    if (str == "hyperelastic_mooney_rivlin" || str == "mooney_rivlin") 
        return MaterialModel::HYPERELASTIC_MOONEY_RIVLIN;
    throw std::invalid_argument("Unknown material model: " + str);
}

// ============================================================================
// Helper Functions
// ============================================================================

double compute_von_mises_stress(const SymmetricTensor<2, 3>& stress) {
    // σ_vm = sqrt(3/2 * s:s) where s is deviatoric stress
    SymmetricTensor<2, 3> dev = compute_deviatoric_stress(stress);
    double s_squared = dev * dev;  // Double contraction
    return std::sqrt(1.5 * s_squared);
}

std::array<double, 3> compute_principal_stresses(const SymmetricTensor<2, 3>& stress) {
    auto eigenvalues = eigenvalues(stress);
    std::array<double, 3> principals = {eigenvalues[0], eigenvalues[1], eigenvalues[2]};
    std::sort(principals.begin(), principals.end(), std::greater<double>());
    return principals;
}

double compute_hydrostatic_stress(const SymmetricTensor<2, 3>& stress) {
    return trace(stress) / 3.0;
}

SymmetricTensor<2, 3> compute_deviatoric_stress(const SymmetricTensor<2, 3>& stress) {
    double p = compute_hydrostatic_stress(stress);
    SymmetricTensor<2, 3> dev = stress;
    for (unsigned int i = 0; i < 3; ++i) {
        dev[i][i] -= p;
    }
    return dev;
}

// ============================================================================
// IsotropicElasticProperties Implementation
// ============================================================================

SymmetricTensor<4, 3> IsotropicElasticProperties::get_elasticity_tensor() const {
    SymmetricTensor<4, 3> C;
    
    const double lam = lambda();
    const double mu_ = mu();
    
    // Build C_ijkl = λ*δ_ij*δ_kl + μ*(δ_ik*δ_jl + δ_il*δ_jk)
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
                    
                    C[i][j][k][l] = lam * delta_ij * delta_kl
                                  + mu_ * (delta_ik * delta_jl + delta_il * delta_jk);
                }
            }
        }
    }
    
    return C;
}

SymmetricTensor<2, 3> IsotropicElasticProperties::get_thermal_expansion_tensor() const {
    SymmetricTensor<2, 3> alpha;
    for (unsigned int i = 0; i < 3; ++i) {
        alpha[i][i] = thermal_expansion_coeff;
    }
    return alpha;
}

// ============================================================================
// OrthotropicElasticProperties Implementation
// ============================================================================

OrthotropicElasticProperties::OrthotropicElasticProperties()
    : E1(0), E2(0), E3(0),
      nu12(0), nu13(0), nu23(0),
      G12(0), G13(0), G23(0),
      density(0),
      alpha1(0), alpha2(0), alpha3(0) {
    // Initialize orientation to identity
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            orientation_matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

SymmetricTensor<4, 3> OrthotropicElasticProperties::get_compliance_tensor() const {
    // Compute derived Poisson's ratios from symmetry
    const double nu21 = nu12 * E2 / E1;
    const double nu31 = nu13 * E3 / E1;
    const double nu32 = nu23 * E3 / E2;
    
    // Build 6x6 compliance matrix in Voigt notation
    // [S] such that {ε} = [S]{σ}
    // Order: [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]
    
    FullMatrix<double> S(6, 6);
    S = 0;
    
    // Normal components
    S(0, 0) = 1.0 / E1;
    S(1, 1) = 1.0 / E2;
    S(2, 2) = 1.0 / E3;
    
    // Coupling terms
    S(0, 1) = S(1, 0) = -nu12 / E1;
    S(0, 2) = S(2, 0) = -nu13 / E1;
    S(1, 2) = S(2, 1) = -nu23 / E2;
    
    // Shear components (factor of 2 for engineering shear strain)
    S(3, 3) = 1.0 / G23;
    S(4, 4) = 1.0 / G13;
    S(5, 5) = 1.0 / G12;
    
    // Convert to 4th order tensor
    SymmetricTensor<4, 3> S_tensor;
    
    // Mapping from tensor indices to Voigt indices
    auto voigt_index = [](unsigned int i, unsigned int j) -> unsigned int {
        if (i == j) return i;
        if ((i == 1 && j == 2) || (i == 2 && j == 1)) return 3;
        if ((i == 0 && j == 2) || (i == 2 && j == 0)) return 4;
        if ((i == 0 && j == 1) || (i == 1 && j == 0)) return 5;
        return 0;
    };
    
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {
                for (unsigned int l = 0; l < 3; ++l) {
                    unsigned int I = voigt_index(i, j);
                    unsigned int J = voigt_index(k, l);
                    
                    double factor = 1.0;
                    // Engineering strain factors
                    if (I > 2) factor *= 2.0;
                    if (J > 2) factor *= 2.0;
                    
                    S_tensor[i][j][k][l] = S(I, J) / factor;
                }
            }
        }
    }
    
    return S_tensor;
}

SymmetricTensor<4, 3> OrthotropicElasticProperties::get_elasticity_tensor() const {
    // Compute [C] = [S]^(-1) by inverting the compliance matrix
    
    const double nu21 = nu12 * E2 / E1;
    const double nu31 = nu13 * E3 / E1;
    const double nu32 = nu23 * E3 / E2;
    
    // Determinant of compliance matrix normal block
    double Delta = 1.0 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2.0*nu12*nu23*nu31;
    
    if (std::abs(Delta) < 1e-15) {
        throw std::runtime_error("Orthotropic compliance matrix is singular");
    }
    
    // Compute stiffness components
    double C11 = E1 * (1.0 - nu23*nu32) / Delta;
    double C22 = E2 * (1.0 - nu13*nu31) / Delta;
    double C33 = E3 * (1.0 - nu12*nu21) / Delta;
    double C12 = E1 * (nu21 + nu31*nu23) / Delta;
    double C13 = E1 * (nu31 + nu21*nu32) / Delta;
    double C23 = E2 * (nu32 + nu12*nu31) / Delta;
    double C44 = G23;
    double C55 = G13;
    double C66 = G12;
    
    // Build 4th order tensor
    SymmetricTensor<4, 3> C;
    C = 0;
    
    // Normal components
    C[0][0][0][0] = C11;
    C[1][1][1][1] = C22;
    C[2][2][2][2] = C33;
    
    // Coupling components
    C[0][0][1][1] = C[1][1][0][0] = C12;
    C[0][0][2][2] = C[2][2][0][0] = C13;
    C[1][1][2][2] = C[2][2][1][1] = C23;
    
    // Shear components
    C[1][2][1][2] = C[2][1][2][1] = C[1][2][2][1] = C[2][1][1][2] = C44 / 2.0;
    C[0][2][0][2] = C[2][0][2][0] = C[0][2][2][0] = C[2][0][0][2] = C55 / 2.0;
    C[0][1][0][1] = C[1][0][1][0] = C[0][1][1][0] = C[1][0][0][1] = C66 / 2.0;
    
    return C;
}

SymmetricTensor<4, 3> OrthotropicElasticProperties::get_rotated_elasticity_tensor() const {
    // Get elasticity tensor in material coordinates
    SymmetricTensor<4, 3> C_mat = get_elasticity_tensor();
    
    // Rotate to global coordinates: C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs
    SymmetricTensor<4, 3> C_global;
    C_global = 0;
    
    const Tensor<2, 3>& R = orientation_matrix;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                    for (unsigned int p = 0; p < 3; ++p)
                        for (unsigned int q = 0; q < 3; ++q)
                            for (unsigned int r = 0; r < 3; ++r)
                                for (unsigned int s = 0; s < 3; ++s)
                                    C_global[i][j][k][l] += 
                                        R[i][p] * R[j][q] * R[k][r] * R[l][s] * C_mat[p][q][r][s];
    
    return C_global;
}

SymmetricTensor<2, 3> OrthotropicElasticProperties::get_thermal_expansion_tensor() const {
    // Thermal expansion in material coordinates
    SymmetricTensor<2, 3> alpha_mat;
    alpha_mat = 0;
    alpha_mat[0][0] = alpha1;
    alpha_mat[1][1] = alpha2;
    alpha_mat[2][2] = alpha3;
    
    // Rotate to global: α'_ij = R_ip R_jq α_pq
    SymmetricTensor<2, 3> alpha_global;
    alpha_global = 0;
    
    const Tensor<2, 3>& R = orientation_matrix;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    alpha_global[i][j] += R[i][p] * R[j][q] * alpha_mat[p][q];
    
    return alpha_global;
}

bool OrthotropicElasticProperties::is_thermodynamically_valid() const {
    // The compliance matrix must be positive definite
    // This requires checking several inequalities
    
    if (E1 <= 0 || E2 <= 0 || E3 <= 0) return false;
    if (G12 <= 0 || G13 <= 0 || G23 <= 0) return false;
    
    // Poisson ratio bounds
    const double nu21 = nu12 * E2 / E1;
    const double nu31 = nu13 * E3 / E1;
    const double nu32 = nu23 * E3 / E2;
    
    // |nu_ij| < sqrt(E_i / E_j)
    if (std::abs(nu12) >= std::sqrt(E1 / E2)) return false;
    if (std::abs(nu13) >= std::sqrt(E1 / E3)) return false;
    if (std::abs(nu23) >= std::sqrt(E2 / E3)) return false;
    
    // Determinant condition
    double det = 1.0 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2.0*nu21*nu32*nu13;
    if (det <= 0) return false;
    
    return true;
}

void OrthotropicElasticProperties::set_orientation_euler(
    double phi, double theta, double psi) {
    // Z-X-Z Euler angles
    const double c1 = std::cos(phi), s1 = std::sin(phi);
    const double c2 = std::cos(theta), s2 = std::sin(theta);
    const double c3 = std::cos(psi), s3 = std::sin(psi);
    
    orientation_matrix[0][0] = c1*c3 - c2*s1*s3;
    orientation_matrix[0][1] = -c1*s3 - c2*c3*s1;
    orientation_matrix[0][2] = s1*s2;
    orientation_matrix[1][0] = c3*s1 + c1*c2*s3;
    orientation_matrix[1][1] = c1*c2*c3 - s1*s3;
    orientation_matrix[1][2] = -c1*s2;
    orientation_matrix[2][0] = s2*s3;
    orientation_matrix[2][1] = c3*s2;
    orientation_matrix[2][2] = c2;
}

void OrthotropicElasticProperties::set_orientation_from_fiber_direction(
    const Tensor<1, 3>& fiber_dir) {
    // Set material 1-direction along fiber, construct orthonormal basis
    Tensor<1, 3> e1 = fiber_dir / fiber_dir.norm();
    
    // Find a vector not parallel to e1
    Tensor<1, 3> temp;
    if (std::abs(e1[0]) < 0.9) {
        temp[0] = 1.0; temp[1] = 0.0; temp[2] = 0.0;
    } else {
        temp[0] = 0.0; temp[1] = 1.0; temp[2] = 0.0;
    }
    
    // Gram-Schmidt to get e2
    double proj = temp * e1;
    Tensor<1, 3> e2;
    for (unsigned int i = 0; i < 3; ++i) {
        e2[i] = temp[i] - proj * e1[i];
    }
    e2 /= e2.norm();
    
    // e3 = e1 × e2
    Tensor<1, 3> e3;
    e3[0] = e1[1]*e2[2] - e1[2]*e2[1];
    e3[1] = e1[2]*e2[0] - e1[0]*e2[2];
    e3[2] = e1[0]*e2[1] - e1[1]*e2[0];
    
    // Build rotation matrix (columns are basis vectors)
    for (unsigned int i = 0; i < 3; ++i) {
        orientation_matrix[i][0] = e1[i];
        orientation_matrix[i][1] = e2[i];
        orientation_matrix[i][2] = e3[i];
    }
}

// ============================================================================
// ElastoplasticVonMisesProperties Implementation
// ============================================================================

ElastoplasticVonMisesProperties::ElastoplasticVonMisesProperties()
    : youngs_modulus(0), poissons_ratio(0), density(0),
      initial_yield_stress(0),
      hardening_type(HardeningType::LINEAR),
      isotropic_hardening_modulus(0),
      kinematic_hardening_modulus(0),
      power_law_exponent(0),
      power_law_reference_strain(1.0),
      thermal_expansion_coeff(0) {}

double ElastoplasticVonMisesProperties::get_yield_stress(
    double equiv_plastic_strain) const {
    
    switch (hardening_type) {
        case HardeningType::PERFECT_PLASTIC:
            return initial_yield_stress;
            
        case HardeningType::LINEAR:
            return initial_yield_stress + 
                   isotropic_hardening_modulus * equiv_plastic_strain;
            
        case HardeningType::POWER_LAW:
            return initial_yield_stress * 
                   std::pow(1.0 + equiv_plastic_strain / power_law_reference_strain,
                           power_law_exponent);
            
        case HardeningType::TABULAR: {
            if (hardening_curve.empty())
                return initial_yield_stress;
            
            // Linear interpolation in hardening curve
            if (equiv_plastic_strain <= hardening_curve.front().first)
                return hardening_curve.front().second;
            if (equiv_plastic_strain >= hardening_curve.back().first)
                return hardening_curve.back().second;
            
            for (size_t i = 1; i < hardening_curve.size(); ++i) {
                if (equiv_plastic_strain <= hardening_curve[i].first) {
                    double t = (equiv_plastic_strain - hardening_curve[i-1].first) /
                              (hardening_curve[i].first - hardening_curve[i-1].first);
                    return (1.0-t) * hardening_curve[i-1].second + 
                           t * hardening_curve[i].second;
                }
            }
            return hardening_curve.back().second;
        }
        
        default:
            return initial_yield_stress;
    }
}

double ElastoplasticVonMisesProperties::get_hardening_modulus(
    double equiv_plastic_strain) const {
    
    switch (hardening_type) {
        case HardeningType::PERFECT_PLASTIC:
            return 0.0;
            
        case HardeningType::LINEAR:
            return isotropic_hardening_modulus;
            
        case HardeningType::POWER_LAW: {
            if (power_law_reference_strain <= 0) return 0.0;
            return initial_yield_stress * power_law_exponent / power_law_reference_strain *
                   std::pow(1.0 + equiv_plastic_strain / power_law_reference_strain,
                           power_law_exponent - 1.0);
        }
            
        case HardeningType::TABULAR: {
            // Numerical derivative using finite difference
            const double delta = 1e-8;
            return (get_yield_stress(equiv_plastic_strain + delta) -
                   get_yield_stress(equiv_plastic_strain)) / delta;
        }
        
        default:
            return 0.0;
    }
}

SymmetricTensor<4, 3> ElastoplasticVonMisesProperties::get_elastic_tensor() const {
    IsotropicElasticProperties iso(youngs_modulus, poissons_ratio, density);
    return iso.get_elasticity_tensor();
}

SymmetricTensor<2, 3> ElastoplasticVonMisesProperties::get_thermal_expansion_tensor() const {
    SymmetricTensor<2, 3> alpha;
    for (unsigned int i = 0; i < 3; ++i) {
        alpha[i][i] = thermal_expansion_coeff;
    }
    return alpha;
}

bool ElastoplasticVonMisesProperties::is_valid() const {
    return youngs_modulus > 0 &&
           poissons_ratio > -1.0 && poissons_ratio < 0.5 &&
           initial_yield_stress > 0 &&
           density >= 0;
}

// ============================================================================
// HyperelasticNeoHookeanProperties Implementation
// ============================================================================

HyperelasticNeoHookeanProperties 
HyperelasticNeoHookeanProperties::from_engineering_constants(
    double E, double nu, double rho) {
    double mu = E / (2.0 * (1.0 + nu));
    double kappa = E / (3.0 * (1.0 - 2.0 * nu));
    return HyperelasticNeoHookeanProperties(mu, kappa, rho);
}

double HyperelasticNeoHookeanProperties::get_strain_energy(
    const Tensor<2, 3>& F) const {
    
    // Right Cauchy-Green tensor: C = F^T * F
    Tensor<2, 3> C;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j) {
            C[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                C[i][j] += F[k][i] * F[k][j];
        }
    
    // I1 = tr(C)
    double I1 = C[0][0] + C[1][1] + C[2][2];
    
    // J = det(F)
    double J = determinant(F);
    double lnJ = std::log(J);
    
    // Lamé parameters
    double mu_ = shear_modulus;
    double lambda = bulk_modulus - 2.0/3.0 * mu_;
    
    // W = (μ/2)(I1 - 3) - μ*ln(J) + (λ/2)(ln(J))²
    return 0.5 * mu_ * (I1 - 3.0) - mu_ * lnJ + 0.5 * lambda * lnJ * lnJ;
}

SymmetricTensor<2, 3> HyperelasticNeoHookeanProperties::get_pk2_stress(
    const Tensor<2, 3>& F) const {
    
    // Right Cauchy-Green tensor: C = F^T * F
    SymmetricTensor<2, 3> C;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            C[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                C[i][j] += F[k][i] * F[k][j];
        }
    
    // Determinant J = det(F)
    double J = determinant(F);
    
    // Inverse of C
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    // Lamé parameters
    double mu_ = shear_modulus;
    double lambda = bulk_modulus - 2.0/3.0 * mu_;
    
    // 2nd Piola-Kirchhoff stress for Neo-Hookean:
    // S = μ*(I - C^{-1}) + λ*ln(J)*C^{-1}
    SymmetricTensor<2, 3> I;
    I = 0;
    for (unsigned int i = 0; i < 3; ++i)
        I[i][i] = 1.0;
    
    SymmetricTensor<2, 3> S = mu_ * (I - C_inv) + lambda * std::log(J) * C_inv;
    
    return S;
}

SymmetricTensor<4, 3> HyperelasticNeoHookeanProperties::get_material_tangent(
    const Tensor<2, 3>& F) const {
    
    // C = F^T * F
    SymmetricTensor<2, 3> C;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            C[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                C[i][j] += F[k][i] * F[k][j];
        }
    
    double J = determinant(F);
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    double mu_ = shear_modulus;
    double lambda = bulk_modulus - 2.0/3.0 * mu_;
    
    // Material tangent: 2 * ∂S/∂C = ∂²W/∂C∂C
    // For Neo-Hookean: 
    // CC_IJKL = λ*C^{-1}_IJ*C^{-1}_KL + (μ - λ*ln(J))*(C^{-1}_IK*C^{-1}_JL + C^{-1}_IL*C^{-1}_JK)
    SymmetricTensor<4, 3> tangent;
    tangent = 0;
    
    double factor = mu_ - lambda * std::log(J);
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l) {
                    tangent[i][j][k][l] = 
                        lambda * C_inv[i][j] * C_inv[k][l] +
                        factor * (C_inv[i][k] * C_inv[j][l] + C_inv[i][l] * C_inv[j][k]);
                }
    
    return tangent;
}

SymmetricTensor<2, 3> HyperelasticNeoHookeanProperties::get_cauchy_stress(
    const Tensor<2, 3>& F) const {
    
    SymmetricTensor<2, 3> S = get_pk2_stress(F);
    double J = determinant(F);
    
    // Cauchy stress: σ = (1/J) * F * S * F^T
    Tensor<2, 3> sigma_full;
    sigma_full = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    sigma_full[i][j] += F[i][p] * S[p][q] * F[j][q];
    
    sigma_full *= (1.0 / J);
    
    // Symmetrize
    SymmetricTensor<2, 3> sigma;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
            sigma[i][j] = 0.5 * (sigma_full[i][j] + sigma_full[j][i]);
    
    return sigma;
}

// ============================================================================
// HyperelasticMooneyRivlinProperties Implementation
// ============================================================================

double HyperelasticMooneyRivlinProperties::get_strain_energy(
    const Tensor<2, 3>& F) const {
    
    double J = determinant(F);
    
    // Right Cauchy-Green: C = F^T * F
    Tensor<2, 3> C;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j) {
            C[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                C[i][j] += F[k][i] * F[k][j];
        }
    
    // Deviatoric C: C_bar = J^(-2/3) * C
    double J23 = std::pow(J, -2.0/3.0);
    
    // I1 = tr(C_bar)
    double I1_bar = J23 * (C[0][0] + C[1][1] + C[2][2]);
    
    // I2 = 0.5 * (I1^2 - tr(C^2))
    double trC2 = 0;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            trC2 += C[i][j] * C[j][i];
    
    double I1 = C[0][0] + C[1][1] + C[2][2];
    double I2_bar = J23 * J23 * 0.5 * (I1 * I1 - trC2);
    
    // W = C10*(I1_bar - 3) + C01*(I2_bar - 3) + (κ/2)*(J - 1)²
    return C10 * (I1_bar - 3.0) + C01 * (I2_bar - 3.0) + 
           0.5 * bulk_modulus * (J - 1.0) * (J - 1.0);
}

SymmetricTensor<2, 3> HyperelasticMooneyRivlinProperties::get_pk2_stress(
    const Tensor<2, 3>& F) const {
    
    double J = determinant(F);
    
    // C = F^T * F
    SymmetricTensor<2, 3> C;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            C[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                C[i][j] += F[k][i] * F[k][j];
        }
    
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    // I1 = tr(C)
    double I1 = trace(C);
    
    // Identity tensor
    SymmetricTensor<2, 3> I;
    I = 0;
    for (unsigned int i = 0; i < 3; ++i)
        I[i][i] = 1.0;
    
    // Deviatoric factors
    double J23 = std::pow(J, -2.0/3.0);
    
    // S = 2 * C10 * J^(-2/3) * (I - (1/3)*I1*C^{-1}) 
    //   + 2 * C01 * J^(-4/3) * (I1*I - C - (2/3)*I2*C^{-1})
    //   + κ * J * (J - 1) * C^{-1}
    
    // Simplified form for implementation
    SymmetricTensor<2, 3> S;
    
    // First term: Mooney-Rivlin deviatoric
    double J43 = J23 * J23;
    
    // Use simpler form: treat as Neo-Hookean-like for basic implementation
    double mu_eff = 2.0 * (C10 + C01);
    double lambda_eff = bulk_modulus - 2.0/3.0 * mu_eff;
    
    S = mu_eff * (I - C_inv) + lambda_eff * std::log(J) * C_inv;
    
    // Add volumetric correction
    S += bulk_modulus * J * (J - 1.0) * C_inv;
    
    return S;
}

SymmetricTensor<4, 3> HyperelasticMooneyRivlinProperties::get_material_tangent(
    const Tensor<2, 3>& F) const {
    
    // Approximate tangent using effective properties
    double J = determinant(F);
    
    SymmetricTensor<2, 3> C;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j) {
            C[i][j] = 0;
            for (unsigned int k = 0; k < 3; ++k)
                C[i][j] += F[k][i] * F[k][j];
        }
    
    SymmetricTensor<2, 3> C_inv = invert(C);
    
    double mu_eff = 2.0 * (C10 + C01);
    double lambda_eff = bulk_modulus - 2.0/3.0 * mu_eff;
    
    SymmetricTensor<4, 3> tangent;
    tangent = 0;
    
    double factor = mu_eff - lambda_eff * std::log(J);
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l) {
                    tangent[i][j][k][l] = 
                        lambda_eff * C_inv[i][j] * C_inv[k][l] +
                        factor * (C_inv[i][k] * C_inv[j][l] + C_inv[i][l] * C_inv[j][k]);
                }
    
    return tangent;
}

SymmetricTensor<2, 3> HyperelasticMooneyRivlinProperties::get_cauchy_stress(
    const Tensor<2, 3>& F) const {
    
    SymmetricTensor<2, 3> S = get_pk2_stress(F);
    double J = determinant(F);
    
    Tensor<2, 3> sigma_full;
    sigma_full = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int p = 0; p < 3; ++p)
                for (unsigned int q = 0; q < 3; ++q)
                    sigma_full[i][j] += F[i][p] * S[p][q] * F[j][q];
    
    sigma_full *= (1.0 / J);
    
    SymmetricTensor<2, 3> sigma;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
            sigma[i][j] = 0.5 * (sigma_full[i][j] + sigma_full[j][i]);
    
    return sigma;
}

// ============================================================================
// Material Implementation
// ============================================================================

double Material::get_density() const {
    return std::visit([](const auto& props) -> double {
        using T = std::decay_t<decltype(props)>;
        if constexpr (std::is_same_v<T, IsotropicElasticProperties>)
            return props.density;
        else if constexpr (std::is_same_v<T, OrthotropicElasticProperties>)
            return props.density;
        else if constexpr (std::is_same_v<T, ElastoplasticVonMisesProperties>)
            return props.density;
        else if constexpr (std::is_same_v<T, HyperelasticNeoHookeanProperties>)
            return props.density;
        else if constexpr (std::is_same_v<T, HyperelasticMooneyRivlinProperties>)
            return props.density;
        else
            return 0.0;
    }, properties);
}

double Material::get_thermal_expansion() const {
    return std::visit([](const auto& props) -> double {
        using T = std::decay_t<decltype(props)>;
        if constexpr (std::is_same_v<T, IsotropicElasticProperties>)
            return props.thermal_expansion_coeff;
        else if constexpr (std::is_same_v<T, ElastoplasticVonMisesProperties>)
            return props.thermal_expansion_coeff;
        else if constexpr (std::is_same_v<T, OrthotropicElasticProperties>)
            return (props.alpha1 + props.alpha2 + props.alpha3) / 3.0;
        else
            return 0.0;
    }, properties);
}

SymmetricTensor<4, 3> Material::get_elasticity_tensor() const {
    return std::visit([](const auto& props) -> SymmetricTensor<4, 3> {
        using T = std::decay_t<decltype(props)>;
        if constexpr (std::is_same_v<T, IsotropicElasticProperties>)
            return props.get_elasticity_tensor();
        else if constexpr (std::is_same_v<T, OrthotropicElasticProperties>)
            return props.get_rotated_elasticity_tensor();
        else if constexpr (std::is_same_v<T, ElastoplasticVonMisesProperties>)
            return props.get_elastic_tensor();
        else if constexpr (std::is_same_v<T, HyperelasticNeoHookeanProperties>) {
            // Return linearized tensor at F = I
            IsotropicElasticProperties iso;
            iso.youngs_modulus = 3.0 * props.bulk_modulus * (1.0 - 2.0 * 0.3) / (1.0 + 0.3);
            iso.poissons_ratio = 0.3;
            return iso.get_elasticity_tensor();
        }
        else if constexpr (std::is_same_v<T, HyperelasticMooneyRivlinProperties>) {
            double mu = props.initial_shear_modulus();
            double E = 9.0 * props.bulk_modulus * mu / (3.0 * props.bulk_modulus + mu);
            double nu = (3.0 * props.bulk_modulus - 2.0 * mu) / (6.0 * props.bulk_modulus + 2.0 * mu);
            IsotropicElasticProperties iso(E, nu, props.density);
            return iso.get_elasticity_tensor();
        }
        else {
            return SymmetricTensor<4, 3>();
        }
    }, properties);
}

// ============================================================================
// MaterialLibrary Implementation
// ============================================================================

// Static member definitions
const std::string MaterialLibrary::STEEL_STRUCTURAL = "steel_structural";
const std::string MaterialLibrary::STEEL_STAINLESS_304 = "steel_stainless_304";
const std::string MaterialLibrary::STEEL_STAINLESS_316 = "steel_stainless_316";
const std::string MaterialLibrary::STEEL_4340 = "steel_4340";
const std::string MaterialLibrary::ALUMINUM_6061_T6 = "aluminum_6061_t6";
const std::string MaterialLibrary::ALUMINUM_7075_T6 = "aluminum_7075_t6";
const std::string MaterialLibrary::ALUMINUM_2024_T3 = "aluminum_2024_t3";
const std::string MaterialLibrary::TITANIUM_TI6AL4V = "titanium_ti6al4v";
const std::string MaterialLibrary::COPPER_C10100 = "copper_c10100";
const std::string MaterialLibrary::BRASS_C36000 = "brass_c36000";
const std::string MaterialLibrary::CONCRETE_NORMAL = "concrete_normal";
const std::string MaterialLibrary::CONCRETE_HIGH_STRENGTH = "concrete_high_strength";
const std::string MaterialLibrary::CFRP_UNIDIRECTIONAL = "cfrp_unidirectional";
const std::string MaterialLibrary::CFRP_WOVEN = "cfrp_woven";
const std::string MaterialLibrary::GFRP_UNIDIRECTIONAL = "gfrp_unidirectional";
const std::string MaterialLibrary::GFRP_WOVEN = "gfrp_woven";
const std::string MaterialLibrary::WOOD_OAK = "wood_oak";
const std::string MaterialLibrary::WOOD_PINE = "wood_pine";
const std::string MaterialLibrary::RUBBER_NATURAL = "rubber_natural";
const std::string MaterialLibrary::RUBBER_NEOPRENE = "rubber_neoprene";
const std::string MaterialLibrary::SILICONE = "silicone";

MaterialLibrary::MaterialLibrary() {
    populate_preset_materials();
}

void MaterialLibrary::add_material(const Material& mat) {
    materials_[mat.id] = mat;
}

const Material& MaterialLibrary::get_material(const std::string& id) const {
    auto it = materials_.find(id);
    if (it == materials_.end())
        throw std::runtime_error("Material not found: " + id);
    return it->second;
}

bool MaterialLibrary::has_material(const std::string& id) const {
    return materials_.find(id) != materials_.end();
}

std::vector<std::string> MaterialLibrary::list_materials() const {
    std::vector<std::string> ids;
    ids.reserve(materials_.size());
    for (const auto& [id, mat] : materials_)
        ids.push_back(id);
    return ids;
}

std::vector<Material> MaterialLibrary::get_all_materials() const {
    std::vector<Material> mats;
    mats.reserve(materials_.size());
    for (const auto& [id, mat] : materials_)
        mats.push_back(mat);
    return mats;
}

void MaterialLibrary::remove_material(const std::string& id) {
    materials_.erase(id);
}

void MaterialLibrary::clear() {
    materials_.clear();
}

void MaterialLibrary::reload_presets() {
    clear();
    populate_preset_materials();
}

void MaterialLibrary::populate_preset_materials() {
    add_steel_presets();
    add_aluminum_presets();
    add_other_metal_presets();
    add_composite_presets();
    add_rubber_presets();
    add_other_presets();
}

void MaterialLibrary::add_steel_presets() {
    // ASTM A36 Structural Steel
    {
        Material mat;
        mat.id = STEEL_STRUCTURAL;
        mat.name = "Structural Steel (ASTM A36)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(200e9, 0.3, 7850, 12e-6);
        mat.yield_strength = 250e6;
        mat.ultimate_strength = 400e6;
        add_material(mat);
    }
    
    // 304 Stainless Steel
    {
        Material mat;
        mat.id = STEEL_STAINLESS_304;
        mat.name = "Stainless Steel 304";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(193e9, 0.29, 8000, 17.3e-6);
        mat.yield_strength = 215e6;
        mat.ultimate_strength = 505e6;
        add_material(mat);
    }
    
    // 316 Stainless Steel
    {
        Material mat;
        mat.id = STEEL_STAINLESS_316;
        mat.name = "Stainless Steel 316";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(193e9, 0.3, 8000, 16e-6);
        mat.yield_strength = 290e6;
        mat.ultimate_strength = 580e6;
        add_material(mat);
    }
    
    // AISI 4340 Steel
    {
        Material mat;
        mat.id = STEEL_4340;
        mat.name = "AISI 4340 Steel";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(205e9, 0.29, 7850, 12.3e-6);
        mat.yield_strength = 470e6;
        mat.ultimate_strength = 745e6;
        add_material(mat);
    }
}

void MaterialLibrary::add_aluminum_presets() {
    // Aluminum 6061-T6
    {
        Material mat;
        mat.id = ALUMINUM_6061_T6;
        mat.name = "Aluminum 6061-T6";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(68.9e9, 0.33, 2700, 23.6e-6);
        mat.yield_strength = 276e6;
        mat.ultimate_strength = 310e6;
        add_material(mat);
    }
    
    // Aluminum 7075-T6
    {
        Material mat;
        mat.id = ALUMINUM_7075_T6;
        mat.name = "Aluminum 7075-T6";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(71.7e9, 0.33, 2810, 23.4e-6);
        mat.yield_strength = 503e6;
        mat.ultimate_strength = 572e6;
        add_material(mat);
    }
    
    // Aluminum 2024-T3
    {
        Material mat;
        mat.id = ALUMINUM_2024_T3;
        mat.name = "Aluminum 2024-T3";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(73.1e9, 0.33, 2780, 23.2e-6);
        mat.yield_strength = 345e6;
        mat.ultimate_strength = 483e6;
        add_material(mat);
    }
}

void MaterialLibrary::add_other_metal_presets() {
    // Titanium Ti-6Al-4V
    {
        Material mat;
        mat.id = TITANIUM_TI6AL4V;
        mat.name = "Titanium Ti-6Al-4V";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(113.8e9, 0.342, 4430, 8.6e-6);
        mat.yield_strength = 880e6;
        mat.ultimate_strength = 950e6;
        add_material(mat);
    }
    
    // Copper C10100
    {
        Material mat;
        mat.id = COPPER_C10100;
        mat.name = "Copper C10100 (OFHC)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(117e9, 0.34, 8940, 17e-6);
        mat.yield_strength = 69e6;
        mat.ultimate_strength = 220e6;
        add_material(mat);
    }
    
    // Brass C36000
    {
        Material mat;
        mat.id = BRASS_C36000;
        mat.name = "Brass C36000 (Free-Cutting)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(97e9, 0.34, 8500, 20.5e-6);
        mat.yield_strength = 124e6;
        mat.ultimate_strength = 338e6;
        add_material(mat);
    }
}

void MaterialLibrary::add_composite_presets() {
    // CFRP Unidirectional (T300/914C typical)
    {
        Material mat;
        mat.id = CFRP_UNIDIRECTIONAL;
        mat.name = "CFRP Unidirectional (T300 type)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC;
        
        OrthotropicElasticProperties props;
        props.E1 = 135e9;   props.E2 = 10e9;    props.E3 = 10e9;
        props.nu12 = 0.3;   props.nu13 = 0.3;   props.nu23 = 0.4;
        props.G12 = 5e9;    props.G13 = 5e9;    props.G23 = 3.5e9;
        props.density = 1600;
        props.alpha1 = -0.3e-6; props.alpha2 = 28e-6; props.alpha3 = 28e-6;
        
        mat.properties = props;
        mat.yield_strength = 1500e6;  // Tensile strength in fiber direction
        mat.ultimate_strength = 1800e6;
        add_material(mat);
    }
    
    // CFRP Woven
    {
        Material mat;
        mat.id = CFRP_WOVEN;
        mat.name = "CFRP Woven (Plain Weave)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC;
        
        OrthotropicElasticProperties props;
        props.E1 = 70e9;    props.E2 = 70e9;    props.E3 = 10e9;
        props.nu12 = 0.1;   props.nu13 = 0.3;   props.nu23 = 0.3;
        props.G12 = 5e9;    props.G13 = 4e9;    props.G23 = 4e9;
        props.density = 1550;
        props.alpha1 = 2e-6; props.alpha2 = 2e-6; props.alpha3 = 30e-6;
        
        mat.properties = props;
        mat.yield_strength = 600e6;
        add_material(mat);
    }
    
    // GFRP Unidirectional
    {
        Material mat;
        mat.id = GFRP_UNIDIRECTIONAL;
        mat.name = "GFRP Unidirectional (E-glass/Epoxy)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC;
        
        OrthotropicElasticProperties props;
        props.E1 = 45e9;    props.E2 = 10e9;    props.E3 = 10e9;
        props.nu12 = 0.3;   props.nu13 = 0.3;   props.nu23 = 0.4;
        props.G12 = 5e9;    props.G13 = 5e9;    props.G23 = 3.5e9;
        props.density = 2000;
        props.alpha1 = 7e-6; props.alpha2 = 21e-6; props.alpha3 = 21e-6;
        
        mat.properties = props;
        mat.yield_strength = 1000e6;
        add_material(mat);
    }
    
    // GFRP Woven (E-glass/Epoxy)
    {
        Material mat;
        mat.id = GFRP_WOVEN;
        mat.name = "GFRP Woven (E-glass/Epoxy)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC;
        
        OrthotropicElasticProperties props;
        props.E1 = 25e9;    props.E2 = 25e9;    props.E3 = 12e9;
        props.nu12 = 0.12;  props.nu13 = 0.3;   props.nu23 = 0.3;
        props.G12 = 4e9;    props.G13 = 3e9;    props.G23 = 3e9;
        props.density = 1900;
        props.alpha1 = 12e-6; props.alpha2 = 12e-6; props.alpha3 = 25e-6;
        
        mat.properties = props;
        mat.yield_strength = 300e6;
        add_material(mat);
    }
}

void MaterialLibrary::add_rubber_presets() {
    // Natural Rubber
    {
        Material mat;
        mat.id = RUBBER_NATURAL;
        mat.name = "Natural Rubber";
        mat.model = MaterialModel::HYPERELASTIC_NEOHOOKEAN;
        
        // μ ≈ 0.4 MPa, nearly incompressible
        mat.properties = HyperelasticNeoHookeanProperties(0.4e6, 2000e6, 1100);
        add_material(mat);
    }
    
    // Neoprene
    {
        Material mat;
        mat.id = RUBBER_NEOPRENE;
        mat.name = "Neoprene Rubber";
        mat.model = MaterialModel::HYPERELASTIC_NEOHOOKEAN;
        mat.properties = HyperelasticNeoHookeanProperties(1.0e6, 2000e6, 1230);
        add_material(mat);
    }
    
    // Silicone
    {
        Material mat;
        mat.id = SILICONE;
        mat.name = "Silicone Rubber";
        mat.model = MaterialModel::HYPERELASTIC_MOONEY_RIVLIN;
        
        HyperelasticMooneyRivlinProperties props;
        props.C10 = 0.11e6;
        props.C01 = 0.02e6;
        props.bulk_modulus = 1000e6;  // Nearly incompressible
        props.density = 1100;
        
        mat.properties = props;
        add_material(mat);
    }
}

void MaterialLibrary::add_other_presets() {
    // Concrete (normal strength)
    {
        Material mat;
        mat.id = CONCRETE_NORMAL;
        mat.name = "Concrete (Normal Strength)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(30e9, 0.2, 2400, 10e-6);
        mat.ultimate_strength = 30e6;  // Compressive strength
        add_material(mat);
    }
    
    // Concrete (high strength)
    {
        Material mat;
        mat.id = CONCRETE_HIGH_STRENGTH;
        mat.name = "Concrete (High Strength)";
        mat.model = MaterialModel::LINEAR_ELASTIC_ISOTROPIC;
        mat.properties = IsotropicElasticProperties(40e9, 0.2, 2500, 10e-6);
        mat.ultimate_strength = 60e6;
        add_material(mat);
    }
    
    // Wood - Oak (orthotropic)
    {
        Material mat;
        mat.id = WOOD_OAK;
        mat.name = "Oak Wood";
        mat.model = MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC;
        
        OrthotropicElasticProperties props;
        props.E1 = 12.3e9;  props.E2 = 1.0e9;   props.E3 = 0.6e9;
        props.nu12 = 0.35;  props.nu13 = 0.45;  props.nu23 = 0.5;
        props.G12 = 0.7e9;  props.G13 = 0.9e9;  props.G23 = 0.2e9;
        props.density = 750;
        props.alpha1 = 5e-6; props.alpha2 = 30e-6; props.alpha3 = 30e-6;
        
        mat.properties = props;
        mat.yield_strength = 40e6;  // Fiber direction tensile
        add_material(mat);
    }
    
    // Wood - Pine (orthotropic)
    {
        Material mat;
        mat.id = WOOD_PINE;
        mat.name = "Pine Wood";
        mat.model = MaterialModel::LINEAR_ELASTIC_ORTHOTROPIC;
        
        OrthotropicElasticProperties props;
        props.E1 = 9.0e9;   props.E2 = 0.5e9;   props.E3 = 0.3e9;
        props.nu12 = 0.35;  props.nu13 = 0.45;  props.nu23 = 0.5;
        props.G12 = 0.5e9;  props.G13 = 0.6e9;  props.G23 = 0.1e9;
        props.density = 550;
        props.alpha1 = 5e-6; props.alpha2 = 35e-6; props.alpha3 = 35e-6;
        
        mat.properties = props;
        mat.yield_strength = 30e6;
        add_material(mat);
    }
}

} // namespace FEA
