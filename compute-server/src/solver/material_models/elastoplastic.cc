/**
 * @file elastoplastic.cc
 * @brief Implementation of J2 elastoplastic material model
 */

#include "elastoplastic.h"
#include <cmath>
#include <algorithm>

namespace FEA {

// ============================================================================
// Helper Function Implementations
// ============================================================================

SymmetricTensor<4, 3> symmetric_identity_4()
{
    SymmetricTensor<4, 3> I;
    I = 0;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l) {
                    double d_ik = (i == k) ? 1.0 : 0.0;
                    double d_jl = (j == l) ? 1.0 : 0.0;
                    double d_il = (i == l) ? 1.0 : 0.0;
                    double d_jk = (j == k) ? 1.0 : 0.0;
                    I[i][j][k][l] = 0.5 * (d_ik * d_jl + d_il * d_jk);
                }
    
    return I;
}

SymmetricTensor<4, 3> deviatoric_projector()
{
    SymmetricTensor<4, 3> P = symmetric_identity_4();
    
    // Subtract (1/3) I ⊗ I
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l) {
                    double d_ij = (i == j) ? 1.0 : 0.0;
                    double d_kl = (k == l) ? 1.0 : 0.0;
                    P[i][j][k][l] -= d_ij * d_kl / 3.0;
                }
    
    return P;
}

SymmetricTensor<4, 3> outer_product(
    const SymmetricTensor<2, 3>& A,
    const SymmetricTensor<2, 3>& B)
{
    SymmetricTensor<4, 3> result;
    
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                    result[i][j][k][l] = A[i][j] * B[k][l];
    
    return result;
}

// ============================================================================
// ElastoplasticMaterial Implementation
// ============================================================================

ElastoplasticMaterial::ElastoplasticMaterial(
    const ElastoplasticVonMisesProperties& props)
    : E_(props.youngs_modulus),
      nu_(props.poissons_ratio),
      sigma_y0_(props.initial_yield_stress),
      H_(props.isotropic_hardening_modulus),
      C_kin_(props.kinematic_hardening_modulus),
      hardening_type_(props.hardening_type),
      power_n_(props.power_law_exponent),
      eps_0_(props.power_law_reference_strain),
      hardening_curve_(props.hardening_curve),
      alpha_(props.thermal_expansion_coeff)
{
    // Compute derived elastic constants
    mu_ = E_ / (2.0 * (1.0 + nu_));
    K_ = E_ / (3.0 * (1.0 - 2.0 * nu_));
    lambda_ = K_ - 2.0 * mu_ / 3.0;
    
    // Build elastic stiffness tensor
    C_elastic_ = 0;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l) {
                    double d_ij = (i == j) ? 1.0 : 0.0;
                    double d_kl = (k == l) ? 1.0 : 0.0;
                    double d_ik = (i == k) ? 1.0 : 0.0;
                    double d_jl = (j == l) ? 1.0 : 0.0;
                    double d_il = (i == l) ? 1.0 : 0.0;
                    double d_jk = (j == k) ? 1.0 : 0.0;
                    
                    C_elastic_[i][j][k][l] = lambda_ * d_ij * d_kl
                        + mu_ * (d_ik * d_jl + d_il * d_jk);
                }
}

double ElastoplasticMaterial::compute_von_mises(
    const SymmetricTensor<2, 3>& stress)
{
    SymmetricTensor<2, 3> s = deviator(stress);
    return std::sqrt(1.5 * (s * s));
}

double ElastoplasticMaterial::compute_shifted_von_mises(
    const SymmetricTensor<2, 3>& stress,
    const SymmetricTensor<2, 3>& back_stress) const
{
    SymmetricTensor<2, 3> eta = deviator(stress) - back_stress;
    return std::sqrt(1.5 * (eta * eta));
}

double ElastoplasticMaterial::get_yield_stress(double equiv_plastic_strain) const
{
    switch (hardening_type_) {
        case ElastoplasticVonMisesProperties::HardeningType::PERFECT_PLASTIC:
            return sigma_y0_;
            
        case ElastoplasticVonMisesProperties::HardeningType::LINEAR:
            return sigma_y0_ + H_ * equiv_plastic_strain;
            
        case ElastoplasticVonMisesProperties::HardeningType::POWER_LAW:
            return sigma_y0_ * std::pow(1.0 + equiv_plastic_strain / eps_0_, power_n_);
            
        case ElastoplasticVonMisesProperties::HardeningType::TABULAR: {
            if (hardening_curve_.empty()) return sigma_y0_;
            
            if (equiv_plastic_strain <= hardening_curve_.front().first)
                return hardening_curve_.front().second;
            if (equiv_plastic_strain >= hardening_curve_.back().first)
                return hardening_curve_.back().second;
            
            for (size_t i = 1; i < hardening_curve_.size(); ++i) {
                if (equiv_plastic_strain <= hardening_curve_[i].first) {
                    double t = (equiv_plastic_strain - hardening_curve_[i-1].first) /
                              (hardening_curve_[i].first - hardening_curve_[i-1].first);
                    return (1.0 - t) * hardening_curve_[i-1].second +
                           t * hardening_curve_[i].second;
                }
            }
            return hardening_curve_.back().second;
        }
        
        default:
            return sigma_y0_;
    }
}

double ElastoplasticMaterial::get_hardening_modulus(double equiv_plastic_strain) const
{
    switch (hardening_type_) {
        case ElastoplasticVonMisesProperties::HardeningType::PERFECT_PLASTIC:
            return 0.0;
            
        case ElastoplasticVonMisesProperties::HardeningType::LINEAR:
            return H_;
            
        case ElastoplasticVonMisesProperties::HardeningType::POWER_LAW:
            return sigma_y0_ * power_n_ / eps_0_ *
                   std::pow(1.0 + equiv_plastic_strain / eps_0_, power_n_ - 1.0);
            
        case ElastoplasticVonMisesProperties::HardeningType::TABULAR: {
            const double delta = 1e-8;
            return (get_yield_stress(equiv_plastic_strain + delta) -
                   get_yield_stress(equiv_plastic_strain)) / delta;
        }
        
        default:
            return 0.0;
    }
}

bool ElastoplasticMaterial::compute_stress_update(
    const SymmetricTensor<2, 3>& total_strain,
    PlasticityState& state,
    SymmetricTensor<2, 3>& stress) const
{
    // Compute elastic strain: ε_e = ε_total - ε_p
    SymmetricTensor<2, 3> elastic_strain = total_strain - state.plastic_strain;
    
    // Trial stress: σ_trial = C : ε_e
    SymmetricTensor<2, 3> trial_stress = C_elastic_ * elastic_strain;
    
    // Compute shifted deviatoric stress (for kinematic hardening)
    SymmetricTensor<2, 3> trial_dev = deviator(trial_stress);
    SymmetricTensor<2, 3> eta_trial = trial_dev - state.back_stress;
    double q_trial = std::sqrt(1.5 * (eta_trial * eta_trial));
    
    // Get current yield stress
    double sigma_y = get_yield_stress(state.equiv_plastic_strain);
    
    // Yield function: f = q - σ_y
    double f_trial = q_trial - sigma_y;
    
    if (f_trial <= 0) {
        // Elastic step - trial stress is the solution
        stress = trial_stress;
        state.yielded = false;
        return false;
    }
    
    // Plastic step - perform radial return
    state.yielded = true;
    
    // Flow direction: n = (3/2) * eta / q
    SymmetricTensor<2, 3> flow_direction = (1.5 / q_trial) * eta_trial;
    
    // Solve for plastic multiplier increment Δγ
    // f(Δγ) = q_trial - 3μΔγ - σ_y(ε_p + Δγ) = 0
    double delta_gamma = solve_plastic_multiplier(q_trial, sigma_y, 
                                                   state.equiv_plastic_strain);
    
    // Update plastic strain: Δε_p = Δγ * n
    state.plastic_strain += delta_gamma * flow_direction;
    
    // Update equivalent plastic strain
    state.equiv_plastic_strain += delta_gamma;
    
    // Update back stress (kinematic hardening): α += (2/3)*C*Δγ*n
    if (C_kin_ > 0) {
        state.back_stress += (2.0 / 3.0) * C_kin_ * delta_gamma * flow_direction;
    }
    
    // Compute final stress
    // σ = σ_trial - 2μ * Δγ * n
    stress = trial_stress - 2.0 * mu_ * delta_gamma * flow_direction;
    
    return true;
}

bool ElastoplasticMaterial::compute_stress_with_thermal(
    const SymmetricTensor<2, 3>& total_strain,
    double delta_temperature,
    PlasticityState& state,
    SymmetricTensor<2, 3>& stress) const
{
    // Compute thermal strain
    SymmetricTensor<2, 3> thermal_strain;
    thermal_strain = 0;
    for (unsigned int i = 0; i < 3; ++i) {
        thermal_strain[i][i] = alpha_ * delta_temperature;
    }
    
    // Mechanical strain = total - thermal
    SymmetricTensor<2, 3> mech_strain = total_strain - thermal_strain;
    
    return compute_stress_update(mech_strain, state, stress);
}

double ElastoplasticMaterial::solve_plastic_multiplier(
    double trial_mises,
    double yield_stress,
    double equiv_plastic_strain) const
{
    // Newton-Raphson to solve: f(Δγ) = q_trial - 3μΔγ - σ_y(ε_p + Δγ) = 0
    
    const double tol = 1e-10;
    const int max_iter = 50;
    
    double delta_gamma = 0;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        double eps_p_new = equiv_plastic_strain + delta_gamma;
        double sigma_y_new = get_yield_stress(eps_p_new);
        double H_new = get_hardening_modulus(eps_p_new);
        
        // Residual
        double f = trial_mises - 3.0 * mu_ * delta_gamma - sigma_y_new;
        
        if (std::abs(f) < tol * sigma_y_new) {
            break;
        }
        
        // Derivative: df/dΔγ = -3μ - H
        double df = -3.0 * mu_ - H_new - C_kin_;
        
        // Update
        delta_gamma -= f / df;
        
        // Ensure non-negative
        delta_gamma = std::max(0.0, delta_gamma);
    }
    
    return delta_gamma;
}

SymmetricTensor<4, 3> ElastoplasticMaterial::get_consistent_tangent(
    const SymmetricTensor<2, 3>& strain,
    const PlasticityState& state) const
{
    if (!state.yielded) {
        // Elastic tangent
        return C_elastic_;
    }
    
    // Compute elastic strain
    SymmetricTensor<2, 3> elastic_strain = strain - state.plastic_strain;
    SymmetricTensor<2, 3> stress = C_elastic_ * elastic_strain;
    
    // Deviatoric stress
    SymmetricTensor<2, 3> s = deviator(stress);
    SymmetricTensor<2, 3> eta = s - state.back_stress;
    double q = std::sqrt(1.5 * (eta * eta));
    
    if (q < 1e-12) {
        return C_elastic_;
    }
    
    // Flow direction
    SymmetricTensor<2, 3> n = (1.5 / q) * eta;
    
    // Hardening modulus
    double H = get_hardening_modulus(state.equiv_plastic_strain);
    double H_total = H + C_kin_;
    
    // Consistent tangent modulus
    // C_ep = C_e - (2μ)² / (3μ + H) * (n ⊗ n)
    
    double factor = 4.0 * mu_ * mu_ / (3.0 * mu_ + H_total);
    
    SymmetricTensor<4, 3> C_ep = C_elastic_ - factor * outer_product(n, n);
    
    return C_ep;
}

SymmetricTensor<2, 3> ElastoplasticMaterial::radial_return(
    const SymmetricTensor<2, 3>& trial_stress,
    PlasticityState& state,
    const SymmetricTensor<2, 3>& stress_n) const
{
    // This is a helper for more complex return mapping algorithms
    // For simple radial return, see compute_stress_update
    
    SymmetricTensor<2, 3> s_trial = deviator(trial_stress);
    SymmetricTensor<2, 3> eta_trial = s_trial - state.back_stress;
    double q_trial = std::sqrt(1.5 * (eta_trial * eta_trial));
    
    double sigma_y = get_yield_stress(state.equiv_plastic_strain);
    
    // Plastic multiplier
    double delta_gamma = solve_plastic_multiplier(q_trial, sigma_y,
                                                   state.equiv_plastic_strain);
    
    // Return stress
    SymmetricTensor<2, 3> n = (1.5 / q_trial) * eta_trial;
    
    return trial_stress - 2.0 * mu_ * delta_gamma * n;
}

} // namespace FEA
