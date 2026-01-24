#include "mpc_constraint.h"
#include <deal.II/grid/grid_tools.h>
#include <algorithm>
#include <set>
#include <map>
#include <cmath>

namespace FEA {

// ============================================================================
// MPCManager Implementation
// ============================================================================

void MPCManager::add_equation(const MPCEquation& eq) {
    if (!eq.terms.empty() && eq.is_active) {
        equations_.push_back(eq);
    }
}

void MPCManager::add_equations(const std::vector<MPCEquation>& eqs) {
    for (const auto& eq : eqs) {
        add_equation(eq);
    }
}

void MPCManager::apply_to_constraints(AffineConstraints<double>& constraints) const {
    for (const auto& eq : equations_) {
        if (!eq.is_active || eq.terms.empty()) continue;
        
        // Get dependent DOF
        types::global_dof_index dep_dof = eq.dependent_dof();
        
        // Skip if already constrained
        if (constraints.is_constrained(dep_dof)) continue;
        
        // Normalize equation copy
        MPCEquation normalized = eq;
        normalized.normalize();
        
        // Add constraint line
        constraints.add_line(dep_dof);
        
        // Add independent DOF contributions
        for (size_t i = 1; i < normalized.terms.size(); ++i) {
            const auto& term = normalized.terms[i];
            // coefficient is negated because deal.II expects:
            // dep_dof = sum(coef_i * indep_dof_i) + inhomogeneity
            constraints.add_entry(dep_dof, term.dof_index, -term.coefficient);
        }
        
        constraints.set_inhomogeneity(dep_dof, normalized.inhomogeneity);
    }
}

std::vector<std::pair<size_t, size_t>> MPCManager::check_conflicts() const {
    std::vector<std::pair<size_t, size_t>> conflicts;
    
    for (size_t i = 0; i < equations_.size(); ++i) {
        for (size_t j = i + 1; j < equations_.size(); ++j) {
            if (are_conflicting(equations_[i], equations_[j])) {
                conflicts.emplace_back(i, j);
            }
        }
    }
    
    return conflicts;
}

bool MPCManager::is_constrained(types::global_dof_index dof) const {
    for (const auto& eq : equations_) {
        if (!eq.terms.empty() && eq.terms[0].dof_index == dof) {
            return true;
        }
    }
    return false;
}

std::vector<size_t> MPCManager::get_equations_for_dof(types::global_dof_index dof) const {
    std::vector<size_t> result;
    
    for (size_t i = 0; i < equations_.size(); ++i) {
        for (const auto& term : equations_[i].terms) {
            if (term.dof_index == dof) {
                result.push_back(i);
                break;
            }
        }
    }
    
    return result;
}

void MPCManager::eliminate_redundant() {
    std::vector<bool> keep(equations_.size(), true);
    
    for (size_t i = 0; i < equations_.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < equations_.size(); ++j) {
            if (!keep[j]) continue;
            if (are_redundant(equations_[i], equations_[j])) {
                keep[j] = false;
            }
        }
    }
    
    std::vector<MPCEquation> filtered;
    for (size_t i = 0; i < equations_.size(); ++i) {
        if (keep[i]) {
            filtered.push_back(equations_[i]);
        }
    }
    
    equations_ = std::move(filtered);
}

void MPCManager::optimize_ordering() {
    // Simple heuristic: sort by dependent DOF index
    // This can help reduce fill-in during elimination
    std::sort(equations_.begin(), equations_.end(),
        [](const MPCEquation& a, const MPCEquation& b) {
            return a.dependent_dof() < b.dependent_dof();
        });
}

bool MPCManager::are_redundant(const MPCEquation& eq1, const MPCEquation& eq2) const {
    // Check if eq1 and eq2 constrain the same DOF with same relationship
    if (eq1.dependent_dof() != eq2.dependent_dof()) return false;
    if (eq1.terms.size() != eq2.terms.size()) return false;
    
    // Build map of DOF -> coefficient for eq1
    std::map<types::global_dof_index, double> coefs1;
    for (const auto& term : eq1.terms) {
        coefs1[term.dof_index] = term.coefficient;
    }
    
    // Check if eq2 has same DOFs with proportional coefficients
    double ratio = 0.0;
    bool ratio_set = false;
    
    for (const auto& term : eq2.terms) {
        auto it = coefs1.find(term.dof_index);
        if (it == coefs1.end()) return false;
        
        if (std::abs(it->second) < 1e-14 || std::abs(term.coefficient) < 1e-14) {
            if (std::abs(it->second - term.coefficient) > 1e-14) return false;
        } else {
            double r = term.coefficient / it->second;
            if (!ratio_set) {
                ratio = r;
                ratio_set = true;
            } else if (std::abs(r - ratio) > 1e-10) {
                return false;
            }
        }
    }
    
    // Check inhomogeneity
    if (ratio_set && std::abs(ratio) > 1e-14) {
        return std::abs(eq2.inhomogeneity - ratio * eq1.inhomogeneity) < 1e-10;
    }
    
    return true;
}

bool MPCManager::are_conflicting(const MPCEquation& eq1, const MPCEquation& eq2) const {
    // Equations conflict if they constrain the same DOF differently
    if (eq1.dependent_dof() != eq2.dependent_dof()) return false;
    
    // If both constrain same DOF, check if they're consistent
    return !are_redundant(eq1, eq2);
}

// ============================================================================
// MPC Generation Utilities
// ============================================================================

template <int dim>
std::vector<MPCEquation> generate_periodic_mpc(
    const DoFHandler<dim>& dof_handler,
    unsigned int face_pair_direction,
    const Tensor<1, dim>& periodic_displacement) {
    
    std::vector<MPCEquation> equations;
    
    // Find matching node pairs on opposite faces
    // This is a simplified implementation - full version would use
    // GridTools::collect_periodic_faces and more robust matching
    
    const auto& fe = dof_handler.get_fe();
    std::map<std::array<double, dim-1>, types::global_dof_index> min_face_dofs;
    std::map<std::array<double, dim-1>, types::global_dof_index> max_face_dofs;
    
    double domain_min = std::numeric_limits<double>::max();
    double domain_max = std::numeric_limits<double>::lowest();
    
    // First pass: find domain extent
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            double coord = cell->vertex(v)[face_pair_direction];
            domain_min = std::min(domain_min, coord);
            domain_max = std::max(domain_max, coord);
        }
    }
    
    double tol = (domain_max - domain_min) * 1e-10;
    
    // Second pass: collect face nodes
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
        cell->get_dof_indices(cell_dofs);
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            Point<dim> face_center = cell->face(f)->center();
            
            bool is_min_face = std::abs(face_center[face_pair_direction] - domain_min) < tol;
            bool is_max_face = std::abs(face_center[face_pair_direction] - domain_max) < tol;
            
            if (!is_min_face && !is_max_face) continue;
            
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                unsigned int cell_v = GeometryInfo<dim>::face_to_cell_vertices(f, v);
                Point<dim> vertex = cell->vertex(cell_v);
                
                // Create key from coordinates perpendicular to face direction
                std::array<double, dim-1> key;
                unsigned int key_idx = 0;
                for (unsigned int d = 0; d < dim; ++d) {
                    if (d != face_pair_direction) {
                        key[key_idx++] = vertex[d];
                    }
                }
                
                for (unsigned int d = 0; d < dim; ++d) {
                    types::global_dof_index dof = cell_dofs[fe.component_to_system_index(d, cell_v)];
                    
                    // Store using direction-specific key
                    std::array<double, dim-1> dof_key = key;
                    if (is_min_face) {
                        min_face_dofs[dof_key] = dof;
                    } else {
                        max_face_dofs[dof_key] = dof;
                    }
                }
            }
        }
    }
    
    // Create MPC equations for matching pairs
    for (const auto& [key, min_dof] : min_face_dofs) {
        auto it = max_face_dofs.find(key);
        if (it != max_face_dofs.end()) {
            types::global_dof_index max_dof = it->second;
            
            // u_max = u_min + periodic_displacement
            MPCEquation eq;
            eq.terms.emplace_back(max_dof, 1.0);
            eq.terms.emplace_back(min_dof, -1.0);
            // Determine which component this DOF represents
            // (simplified - assumes DOF ordering)
            eq.inhomogeneity = 0.0;  // Would need proper component lookup
            eq.description = "Periodic BC";
            
            equations.push_back(eq);
        }
    }
    
    return equations;
}

template <int dim>
std::vector<MPCEquation> generate_antisymmetric_mpc(
    const DoFHandler<dim>& dof_handler,
    unsigned int normal_direction) {
    
    std::vector<MPCEquation> equations;
    
    // Similar to periodic, but with sign flip
    // u_n(x) = -u_n(-x) for normal component
    // u_t(x) = u_t(-x) for tangential components
    
    // Placeholder implementation
    (void)dof_handler;
    (void)normal_direction;
    
    return equations;
}

template <int dim>
std::vector<MPCEquation> generate_plane_strain_mpc(
    const DoFHandler<dim>& dof_handler,
    unsigned int out_of_plane_direction) {
    
    std::vector<MPCEquation> equations;
    
    if constexpr (dim == 3) {
        const auto& fe = dof_handler.get_fe();
        
        // Find all DOFs in out-of-plane direction
        std::vector<types::global_dof_index> oop_dofs;
        
        for (const auto& cell : dof_handler.active_cell_iterators()) {
            std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
            cell->get_dof_indices(cell_dofs);
            
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
                types::global_dof_index dof = cell_dofs[
                    fe.component_to_system_index(out_of_plane_direction, v)];
                oop_dofs.push_back(dof);
            }
        }
        
        // Remove duplicates
        std::sort(oop_dofs.begin(), oop_dofs.end());
        oop_dofs.erase(std::unique(oop_dofs.begin(), oop_dofs.end()), oop_dofs.end());
        
        // Constrain all OOP displacements to be equal (to first one)
        if (oop_dofs.size() > 1) {
            for (size_t i = 1; i < oop_dofs.size(); ++i) {
                equations.push_back(MPCEquation::equality(oop_dofs[i], oop_dofs[0]));
            }
        }
    }
    
    return equations;
}

template <int dim>
std::vector<MPCEquation> generate_axisymmetric_mpc(
    const DoFHandler<dim>& dof_handler,
    const Point<dim>& axis_point,
    const Tensor<1, dim>& axis_direction) {
    
    std::vector<MPCEquation> equations;
    
    // For axisymmetric analysis, constrain circumferential displacement to zero
    // u_theta = 0 where theta is the circumferential direction
    
    Tensor<1, dim> axis = axis_direction / axis_direction.norm();
    const auto& fe = dof_handler.get_fe();
    
    std::set<types::global_dof_index> processed;
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
        cell->get_dof_indices(cell_dofs);
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            Point<dim> vertex = cell->vertex(v);
            
            // Compute radial direction
            Tensor<1, dim> r;
            for (unsigned int d = 0; d < dim; ++d) {
                r[d] = vertex[d] - axis_point[d];
            }
            
            // Remove axial component
            double axial_comp = r * axis;
            Tensor<1, dim> radial = r - axial_comp * axis;
            double r_mag = radial.norm();
            
            if (r_mag < 1e-14) continue;  // On axis
            
            Tensor<1, dim> e_r = radial / r_mag;
            Tensor<1, dim> e_theta = cross_product_3d(axis, e_r);
            
            // Get DOFs for this vertex
            std::array<types::global_dof_index, dim> vertex_dofs;
            for (unsigned int d = 0; d < dim; ++d) {
                vertex_dofs[d] = cell_dofs[fe.component_to_system_index(d, v)];
            }
            
            // Skip if already processed
            if (processed.count(vertex_dofs[0])) continue;
            processed.insert(vertex_dofs[0]);
            
            // Create MPC: e_theta · u = 0
            // sum(e_theta[d] * u[d]) = 0
            
            // Find component with largest e_theta coefficient for dependent DOF
            unsigned int dep_comp = 0;
            double max_comp = std::abs(e_theta[0]);
            for (unsigned int d = 1; d < dim; ++d) {
                if (std::abs(e_theta[d]) > max_comp) {
                    max_comp = std::abs(e_theta[d]);
                    dep_comp = d;
                }
            }
            
            if (max_comp < 1e-10) continue;
            
            MPCEquation eq;
            eq.terms.emplace_back(vertex_dofs[dep_comp], e_theta[dep_comp]);
            for (unsigned int d = 0; d < dim; ++d) {
                if (d != dep_comp && std::abs(e_theta[d]) > 1e-14) {
                    eq.terms.emplace_back(vertex_dofs[d], e_theta[d]);
                }
            }
            eq.inhomogeneity = 0.0;
            eq.description = "Axisymmetric constraint";
            eq.normalize();
            
            equations.push_back(eq);
        }
    }
    
    return equations;
}

MPCEquation generate_interpolation_mpc(
    types::global_dof_index dependent_dof,
    const std::vector<types::global_dof_index>& support_dofs,
    const std::vector<double>& interpolation_weights) {
    
    if (support_dofs.size() != interpolation_weights.size()) {
        throw std::runtime_error("Interpolation MPC: size mismatch between DOFs and weights");
    }
    
    MPCEquation eq;
    eq.terms.emplace_back(dependent_dof, 1.0);
    
    for (size_t i = 0; i < support_dofs.size(); ++i) {
        eq.terms.emplace_back(support_dofs[i], -interpolation_weights[i]);
    }
    
    eq.description = "Interpolation constraint";
    return eq;
}

// ============================================================================
// Constraint Compatibility
// ============================================================================

CompatibilityResult check_constraint_compatibility(
    const std::vector<MPCEquation>& equations,
    const AffineConstraints<double>& existing_constraints) {
    
    CompatibilityResult result;
    
    // Check for DOFs that are already constrained
    for (size_t i = 0; i < equations.size(); ++i) {
        const auto& eq = equations[i];
        if (eq.terms.empty()) continue;
        
        types::global_dof_index dep_dof = eq.dependent_dof();
        if (existing_constraints.is_constrained(dep_dof)) {
            result.add_issue("Equation " + std::to_string(i) + 
                           ": dependent DOF " + std::to_string(dep_dof) +
                           " is already constrained");
        }
    }
    
    // Check for conflicts between new equations
    for (size_t i = 0; i < equations.size(); ++i) {
        for (size_t j = i + 1; j < equations.size(); ++j) {
            if (equations[i].dependent_dof() == equations[j].dependent_dof()) {
                // Potential conflict - check if consistent
                bool consistent = true;  // Would need more sophisticated check
                
                if (!consistent) {
                    result.add_issue("Equations " + std::to_string(i) + 
                                   " and " + std::to_string(j) + " conflict");
                    result.conflicting_pairs.emplace_back(i, j);
                }
            }
        }
    }
    
    return result;
}

std::vector<MPCEquation> resolve_constraint_conflicts(
    const std::vector<MPCEquation>& equations,
    const AffineConstraints<double>& existing_constraints,
    double tolerance) {
    
    std::vector<MPCEquation> resolved;
    resolved.reserve(equations.size());
    
    for (const auto& eq : equations) {
        if (eq.terms.empty()) continue;
        
        MPCEquation new_eq = eq;
        
        // Substitute any constrained DOFs
        std::vector<MPCTerm> new_terms;
        new_terms.push_back(new_eq.terms[0]);  // Keep dependent
        
        for (size_t i = 1; i < new_eq.terms.size(); ++i) {
            const auto& term = new_eq.terms[i];
            
            if (existing_constraints.is_constrained(term.dof_index)) {
                // Get constraint entries and substitute
                const auto* entries = existing_constraints.get_constraint_entries(term.dof_index);
                double inhom = existing_constraints.get_inhomogeneity(term.dof_index);
                
                if (entries) {
                    for (const auto& entry : *entries) {
                        new_terms.emplace_back(entry.first, term.coefficient * entry.second);
                    }
                }
                new_eq.inhomogeneity += term.coefficient * inhom;
            } else {
                new_terms.push_back(term);
            }
        }
        
        new_eq.terms = std::move(new_terms);
        
        // Remove near-zero terms
        new_eq.terms.erase(
            std::remove_if(new_eq.terms.begin() + 1, new_eq.terms.end(),
                [tolerance](const MPCTerm& t) {
                    return std::abs(t.coefficient) < tolerance;
                }),
            new_eq.terms.end());
        
        // Check if dependent DOF is now constrained
        if (!existing_constraints.is_constrained(new_eq.dependent_dof())) {
            resolved.push_back(new_eq);
        }
    }
    
    return resolved;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template std::vector<MPCEquation> generate_periodic_mpc<3>(
    const DoFHandler<3>&, unsigned int, const Tensor<1, 3>&);

template std::vector<MPCEquation> generate_antisymmetric_mpc<3>(
    const DoFHandler<3>&, unsigned int);

template std::vector<MPCEquation> generate_plane_strain_mpc<3>(
    const DoFHandler<3>&, unsigned int);

template std::vector<MPCEquation> generate_axisymmetric_mpc<3>(
    const DoFHandler<3>&, const Point<3>&, const Tensor<1, 3>&);

} // namespace FEA
