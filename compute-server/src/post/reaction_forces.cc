#include "reaction_forces.h"
#include <cmath>
#include <algorithm>

namespace FEA {

// ============================================================================
// Constructor
// ============================================================================

template <int dim>
ReactionForceCalculator<dim>::ReactionForceCalculator(
    const DoFHandler<dim>& dh,
    const Mapping<dim>& map,
    const AffineConstraints<double>& cons)
    : dof_handler_(dh)
    , mapping_(map)
    , constraints_(cons)
{}

// ============================================================================
// Main Computation Method
// ============================================================================

template <int dim>
void ReactionForceCalculator<dim>::compute(
    const SparseMatrix<double>& K,
    const Vector<double>& u,
    const Vector<double>& F) {
    
    // Compute reactions: R = K*u - F
    reaction_vector_.reinit(dof_handler_.n_dofs());
    K.vmult(reaction_vector_, u);
    reaction_vector_ -= F;
    
    // Zero out reactions at unconstrained DOFs
    // (only constrained DOFs have meaningful reactions)
    for (unsigned int i = 0; i < reaction_vector_.size(); ++i) {
        if (!constraints_.is_constrained(i)) {
            reaction_vector_(i) = 0;
        }
    }
    
    // Extract nodal reactions from the reaction vector
    const auto& fe = dof_handler_.get_fe();
    std::set<unsigned int> processed_vertices;
    nodal_reactions_.clear();
    
    total_force_ = Tensor<1, dim>();
    total_moment_ = Tensor<1, dim>();
    
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            unsigned int vertex_index = cell->vertex_index(v);
            
            // Skip already processed vertices
            if (processed_vertices.count(vertex_index)) continue;
            
            Tensor<1, dim> node_force;
            std::array<bool, dim> constrained = {};
            bool has_reaction = false;
            
            // Check each displacement DOF at this vertex
            for (unsigned int d = 0; d < dim; ++d) {
                types::global_dof_index dof = dof_indices[v * dim + d];
                if (constraints_.is_constrained(dof)) {
                    node_force[d] = reaction_vector_(dof);
                    constrained[d] = true;
                    has_reaction = true;
                }
            }
            
            if (has_reaction) {
                processed_vertices.insert(vertex_index);
                
                NodalReaction nr;
                nr.node_id = vertex_index;
                nr.location = cell->vertex(v);
                nr.force = node_force;
                nr.constrained_dofs = constrained;
                nodal_reactions_.push_back(nr);
                
                // Accumulate total force
                total_force_ += node_force;
                
                // Accumulate moment about origin
                if constexpr (dim == 3) {
                    Tensor<1, dim> r;
                    for (unsigned int d = 0; d < dim; ++d)
                        r[d] = nr.location[d];
                    total_moment_ += cross_product_3d(r, node_force);
                } else {
                    // 2D: moment = x*Fy - y*Fx (scalar, store in z-component)
                    total_moment_[0] = 0;
                    total_moment_[1] = nr.location[0] * node_force[1] - 
                                       nr.location[1] * node_force[0];
                }
            }
        }
    }
    
    // Group by boundary and compute boundary moments
    categorize_by_boundary();
    compute_boundary_moments();
}

// ============================================================================
// Boundary Categorization
// ============================================================================

template <int dim>
void ReactionForceCalculator<dim>::categorize_by_boundary() {
    boundary_reactions_.clear();
    
    const auto& fe = dof_handler_.get_fe();
    
    // Map vertex indices to boundary IDs they belong to
    std::map<unsigned int, std::set<unsigned int>> vertex_to_boundaries;
    
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            unsigned int bid = cell->face(f)->boundary_id();
            
            // Mark all vertices of this face as belonging to this boundary
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                unsigned int cell_vertex = GeometryInfo<dim>::face_to_cell_vertices(f, v);
                unsigned int vertex_index = cell->vertex_index(cell_vertex);
                vertex_to_boundaries[vertex_index].insert(bid);
            }
        }
    }
    
    // Assign nodal reactions to boundaries
    for (const auto& nr : nodal_reactions_) {
        auto it = vertex_to_boundaries.find(nr.node_id);
        if (it == vertex_to_boundaries.end()) continue;
        
        // Assign to each boundary this node belongs to
        for (unsigned int bid : it->second) {
            auto& br = boundary_reactions_[bid];
            br.boundary_id = bid;
            br.force += nr.force;
            
            // Accumulate centroid (will divide by count later)
            for (unsigned int d = 0; d < dim; ++d)
                br.centroid[d] += nr.location[d];
            br.num_nodes++;
        }
    }
    
    // Finalize centroids
    for (auto& [bid, br] : boundary_reactions_) {
        if (br.num_nodes > 0) {
            for (unsigned int d = 0; d < dim; ++d) {
                br.centroid[d] /= br.num_nodes;
            }
        }
    }
}

// ============================================================================
// Boundary Moment Computation
// ============================================================================

template <int dim>
void ReactionForceCalculator<dim>::compute_boundary_moments() {
    // For each boundary, compute moment about its centroid
    
    // First, collect which nodes belong to which boundary
    std::map<unsigned int, std::set<unsigned int>> boundary_to_nodes;
    
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            unsigned int bid = cell->face(f)->boundary_id();
            
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                unsigned int cell_vertex = GeometryInfo<dim>::face_to_cell_vertices(f, v);
                boundary_to_nodes[bid].insert(cell->vertex_index(cell_vertex));
            }
        }
    }
    
    // Compute moments for each boundary
    for (auto& [bid, br] : boundary_reactions_) {
        br.moment = Tensor<1, dim>();
        
        const auto& nodes = boundary_to_nodes[bid];
        
        for (const auto& nr : nodal_reactions_) {
            if (nodes.find(nr.node_id) == nodes.end()) continue;
            
            Tensor<1, dim> r;
            for (unsigned int d = 0; d < dim; ++d)
                r[d] = nr.location[d] - br.centroid[d];
            
            if constexpr (dim == 3) {
                br.moment += cross_product_3d(r, nr.force);
            } else {
                // 2D moment (scalar in z)
                br.moment[0] = 0;
                br.moment[1] += r[0] * nr.force[1] - r[1] * nr.force[0];
            }
        }
    }
}

// ============================================================================
// Result Accessors
// ============================================================================

template <int dim>
std::vector<typename ReactionForceCalculator<dim>::BoundaryReaction>
ReactionForceCalculator<dim>::get_boundary_reactions() const {
    std::vector<BoundaryReaction> result;
    result.reserve(boundary_reactions_.size());
    for (const auto& [id, br] : boundary_reactions_)
        result.push_back(br);
    return result;
}

template <int dim>
std::vector<typename ReactionForceCalculator<dim>::NodalReaction>
ReactionForceCalculator<dim>::get_nodal_reactions() const {
    return nodal_reactions_;
}

template <int dim>
Tensor<1, dim> ReactionForceCalculator<dim>::get_total_moment(
    const Point<dim>& about) const {
    
    // If about origin, return cached value
    if (about.norm() < 1e-14)
        return total_moment_;
    
    // Recompute moment about the specified point
    Tensor<1, dim> moment;
    for (const auto& nr : nodal_reactions_) {
        Tensor<1, dim> r;
        for (unsigned int d = 0; d < dim; ++d)
            r[d] = nr.location[d] - about[d];
        
        if constexpr (dim == 3) {
            moment += cross_product_3d(r, nr.force);
        } else {
            moment[0] = 0;
            moment[1] += r[0] * nr.force[1] - r[1] * nr.force[0];
        }
    }
    return moment;
}

template <int dim>
typename ReactionForceCalculator<dim>::BoundaryReaction
ReactionForceCalculator<dim>::get_reaction_at_boundary(unsigned int boundary_id) const {
    auto it = boundary_reactions_.find(boundary_id);
    if (it != boundary_reactions_.end())
        return it->second;
    return BoundaryReaction{};
}

// ============================================================================
// Equilibrium Check
// ============================================================================

template <int dim>
typename ReactionForceCalculator<dim>::EquilibriumCheck
ReactionForceCalculator<dim>::check_equilibrium(
    const Vector<double>& F,
    const Point<dim>& moment_center) const {
    
    EquilibriumCheck check;
    check.total_reaction_force = total_force_;
    check.total_reaction_moment = get_total_moment(moment_center);
    
    // Compute total applied force from RHS vector
    check.total_applied_force = Tensor<1, dim>();
    check.total_applied_moment = Tensor<1, dim>();
    
    const auto& fe = dof_handler_.get_fe();
    std::set<unsigned int> processed;
    
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            unsigned int vid = cell->vertex_index(v);
            if (processed.count(vid)) continue;
            processed.insert(vid);
            
            Point<dim> vertex = cell->vertex(v);
            Tensor<1, dim> node_force;
            
            for (unsigned int d = 0; d < dim; ++d) {
                types::global_dof_index dof = dof_indices[v * dim + d];
                if (dof < F.size()) {
                    node_force[d] = F(dof);
                }
            }
            
            check.total_applied_force += node_force;
            
            // Compute moment about center
            if constexpr (dim == 3) {
                Tensor<1, dim> r;
                for (unsigned int d = 0; d < dim; ++d)
                    r[d] = vertex[d] - moment_center[d];
                check.total_applied_moment += cross_product_3d(r, node_force);
            } else {
                Tensor<1, dim> r;
                for (unsigned int d = 0; d < dim; ++d)
                    r[d] = vertex[d] - moment_center[d];
                check.total_applied_moment[0] = 0;
                check.total_applied_moment[1] += r[0] * node_force[1] - r[1] * node_force[0];
            }
        }
    }
    
    // Compute residuals (should sum to zero for equilibrium)
    // Applied + Reaction = 0, so residual = Applied + Reaction
    check.force_residual = check.total_applied_force + check.total_reaction_force;
    check.moment_residual = check.total_applied_moment + check.total_reaction_moment;
    
    // Compute percentage errors
    double applied_force_mag = check.total_applied_force.norm();
    double applied_moment_mag = check.total_applied_moment.norm();
    
    check.force_error_percent = (applied_force_mag > 1e-10) ?
        100.0 * check.force_residual.norm() / applied_force_mag : 0.0;
    check.moment_error_percent = (applied_moment_mag > 1e-10) ?
        100.0 * check.moment_residual.norm() / applied_moment_mag : 0.0;
    
    // Consider balanced if errors are less than 1%
    check.is_balanced = (check.force_error_percent < 1.0 &&
                         check.moment_error_percent < 1.0);
    
    return check;
}

// ============================================================================
// JSON Serialization
// ============================================================================

template <int dim>
json ReactionForceCalculator<dim>::to_json() const {
    json j;
    
    // Boundary reactions
    json boundaries = json::array();
    for (const auto& [id, br] : boundary_reactions_) {
        boundaries.push_back(br.to_json());
    }
    j["boundaries"] = boundaries;
    
    // Total force and moment
    j["total_force"] = std::vector<double>{
        total_force_[0], total_force_[1], 
        dim == 3 ? total_force_[2] : 0.0
    };
    j["total_moment"] = std::vector<double>{
        total_moment_[0], total_moment_[1],
        dim == 3 ? total_moment_[2] : 0.0
    };
    
    return j;
}

// ============================================================================
// Explicit Instantiations
// ============================================================================

template class ReactionForceCalculator<3>;
template class ReactionForceCalculator<2>;

} // namespace FEA
