#ifndef MPC_CONSTRAINT_H
#define MPC_CONSTRAINT_H

/**
 * @file mpc_constraint.h
 * @brief Multi-Point Constraint (MPC) utilities
 * 
 * This header provides utilities for general multi-point constraints.
 * MPCs express linear relationships between DOFs:
 * 
 *   sum(c_i * u_i) = c_0
 * 
 * where c_i are coefficients, u_i are DOF values, and c_0 is a constant.
 * 
 * The constraint types in constraint_base.h (RigidConnection, TiedConnection,
 * DirectionalCoupling, CylindricalCoupling) are specific forms of MPCs.
 * 
 * This file adds:
 * - General MPC equation representation
 * - MPC equation manipulation utilities
 * - Constraint compatibility checking
 * - MPC generation from geometric relationships
 */

#include "constraint_base.h"
#include <deal.II/lac/affine_constraints.h>
#include <functional>

namespace FEA {

// ============================================================================
// General MPC Representation
// ============================================================================

/**
 * @brief Single term in an MPC equation
 */
struct MPCTerm {
    types::global_dof_index dof_index;
    double coefficient;
    
    MPCTerm() : dof_index(numbers::invalid_dof_index), coefficient(0.0) {}
    MPCTerm(types::global_dof_index idx, double coef) 
        : dof_index(idx), coefficient(coef) {}
};

/**
 * @brief Multi-Point Constraint equation
 * 
 * Represents: sum(terms[i].coefficient * u[terms[i].dof_index]) = inhomogeneity
 * 
 * The first term (terms[0]) is typically the "dependent" DOF that
 * will be eliminated from the system.
 */
struct MPCEquation {
    std::vector<MPCTerm> terms;
    double inhomogeneity = 0.0;
    std::string description;
    bool is_active = true;
    
    /**
     * @brief Get the dependent DOF (first term)
     */
    types::global_dof_index dependent_dof() const {
        return terms.empty() ? numbers::invalid_dof_index : terms[0].dof_index;
    }
    
    /**
     * @brief Normalize equation so dependent DOF has coefficient 1.0
     */
    void normalize() {
        if (terms.empty()) return;
        double dep_coef = terms[0].coefficient;
        if (std::abs(dep_coef) < 1e-14) return;
        
        for (auto& term : terms) {
            term.coefficient /= dep_coef;
        }
        inhomogeneity /= dep_coef;
    }
    
    /**
     * @brief Create simple equality constraint: u_a = u_b
     */
    static MPCEquation equality(types::global_dof_index dof_a,
                                 types::global_dof_index dof_b) {
        MPCEquation eq;
        eq.terms.emplace_back(dof_a, 1.0);
        eq.terms.emplace_back(dof_b, -1.0);
        eq.description = "DOF equality";
        return eq;
    }
    
    /**
     * @brief Create prescribed displacement: u_a = value
     */
    static MPCEquation prescribed(types::global_dof_index dof,
                                   double value) {
        MPCEquation eq;
        eq.terms.emplace_back(dof, 1.0);
        eq.inhomogeneity = value;
        eq.description = "Prescribed displacement";
        return eq;
    }
    
    /**
     * @brief Create ratio constraint: u_a = ratio * u_b
     */
    static MPCEquation ratio(types::global_dof_index dof_a,
                              types::global_dof_index dof_b,
                              double ratio) {
        MPCEquation eq;
        eq.terms.emplace_back(dof_a, 1.0);
        eq.terms.emplace_back(dof_b, -ratio);
        eq.description = "DOF ratio";
        return eq;
    }
    
    /**
     * @brief Create linear combination: u_dep = sum(c_i * u_i)
     */
    static MPCEquation linear_combination(
        types::global_dof_index dependent_dof,
        const std::vector<std::pair<types::global_dof_index, double>>& contributions) {
        MPCEquation eq;
        eq.terms.emplace_back(dependent_dof, 1.0);
        for (const auto& [dof, coef] : contributions) {
            eq.terms.emplace_back(dof, -coef);
        }
        eq.description = "Linear combination";
        return eq;
    }
};

// ============================================================================
// MPC Manager
// ============================================================================

/**
 * @brief Manages a set of MPC equations
 */
class MPCManager {
public:
    MPCManager() = default;
    
    /**
     * @brief Add an MPC equation
     */
    void add_equation(const MPCEquation& eq);
    
    /**
     * @brief Add multiple equations
     */
    void add_equations(const std::vector<MPCEquation>& eqs);
    
    /**
     * @brief Clear all equations
     */
    void clear() { equations_.clear(); }
    
    /**
     * @brief Get number of equations
     */
    size_t size() const { return equations_.size(); }
    
    /**
     * @brief Access equation by index
     */
    const MPCEquation& operator[](size_t i) const { return equations_[i]; }
    MPCEquation& operator[](size_t i) { return equations_[i]; }
    
    /**
     * @brief Apply all MPCs to AffineConstraints object
     */
    void apply_to_constraints(AffineConstraints<double>& constraints) const;
    
    /**
     * @brief Check for conflicts between MPCs
     * 
     * Returns list of conflicting equation pairs
     */
    std::vector<std::pair<size_t, size_t>> check_conflicts() const;
    
    /**
     * @brief Check if a DOF is constrained by any MPC
     */
    bool is_constrained(types::global_dof_index dof) const;
    
    /**
     * @brief Get all equations constraining a DOF
     */
    std::vector<size_t> get_equations_for_dof(types::global_dof_index dof) const;
    
    /**
     * @brief Eliminate redundant equations
     */
    void eliminate_redundant();
    
    /**
     * @brief Reorder equations to minimize fill-in
     */
    void optimize_ordering();
    
private:
    std::vector<MPCEquation> equations_;
    
    /**
     * @brief Check if two equations are redundant
     */
    bool are_redundant(const MPCEquation& eq1, const MPCEquation& eq2) const;
    
    /**
     * @brief Check if two equations conflict
     */
    bool are_conflicting(const MPCEquation& eq1, const MPCEquation& eq2) const;
};

// ============================================================================
// MPC Generation Utilities
// ============================================================================

/**
 * @brief Generate MPCs for periodic boundary conditions
 * 
 * Creates constraints that tie displacements on opposite faces:
 * u(x_max) = u(x_min) + periodic_displacement
 */
template <int dim>
std::vector<MPCEquation> generate_periodic_mpc(
    const DoFHandler<dim>& dof_handler,
    unsigned int face_pair_direction,
    const Tensor<1, dim>& periodic_displacement = Tensor<1, dim>());

/**
 * @brief Generate MPCs for antisymmetric boundary conditions
 * 
 * For antisymmetry: u_n(x) = -u_n(-x)
 */
template <int dim>
std::vector<MPCEquation> generate_antisymmetric_mpc(
    const DoFHandler<dim>& dof_handler,
    unsigned int normal_direction);

/**
 * @brief Generate MPCs for plane strain assumption in 3D
 * 
 * Constrains out-of-plane displacement to be constant
 */
template <int dim>
std::vector<MPCEquation> generate_plane_strain_mpc(
    const DoFHandler<dim>& dof_handler,
    unsigned int out_of_plane_direction);

/**
 * @brief Generate MPCs for axisymmetric analysis
 * 
 * Constrains circumferential displacement to zero
 */
template <int dim>
std::vector<MPCEquation> generate_axisymmetric_mpc(
    const DoFHandler<dim>& dof_handler,
    const Point<dim>& axis_point,
    const Tensor<1, dim>& axis_direction);

/**
 * @brief Generate MPCs from displacement interpolation
 * 
 * Creates constraint where a node's displacement is interpolated
 * from surrounding nodes (used for hanging nodes, mesh tying)
 */
MPCEquation generate_interpolation_mpc(
    types::global_dof_index dependent_dof,
    const std::vector<types::global_dof_index>& support_dofs,
    const std::vector<double>& interpolation_weights);

// ============================================================================
// Constraint Compatibility
// ============================================================================

/**
 * @brief Result of constraint compatibility check
 */
struct CompatibilityResult {
    bool is_compatible = true;
    std::vector<std::string> issues;
    std::vector<std::pair<size_t, size_t>> conflicting_pairs;
    
    void add_issue(const std::string& issue) {
        is_compatible = false;
        issues.push_back(issue);
    }
};

/**
 * @brief Check if a set of constraints is compatible
 */
CompatibilityResult check_constraint_compatibility(
    const std::vector<MPCEquation>& equations,
    const AffineConstraints<double>& existing_constraints);

/**
 * @brief Resolve constraint conflicts by modifying equations
 * 
 * Attempts to resolve conflicts by:
 * 1. Substituting constrained DOFs
 * 2. Merging redundant constraints
 * 3. Adjusting near-singular coefficients
 */
std::vector<MPCEquation> resolve_constraint_conflicts(
    const std::vector<MPCEquation>& equations,
    const AffineConstraints<double>& existing_constraints,
    double tolerance = 1e-10);

} // namespace FEA

#endif // MPC_CONSTRAINT_H
