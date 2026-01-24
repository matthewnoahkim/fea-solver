#include "constraint_base.h"
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <algorithm>
#include <cmath>
#include <set>

namespace FEA {

// ============================================================================
// Helper Functions
// ============================================================================

const std::string& get_connection_description(const Connection& conn) {
    return std::visit([](const auto& c) -> const std::string& {
        return c.description;
    }, conn);
}

bool is_spring_connection(const Connection& conn) {
    return std::holds_alternative<SpringToGroundConnection>(conn) ||
           std::holds_alternative<SpringConnection>(conn) ||
           std::holds_alternative<BushingConnection>(conn);
}

bool is_constraint_connection(const Connection& conn) {
    return std::holds_alternative<RigidConnection>(conn) ||
           std::holds_alternative<TiedConnection>(conn) ||
           std::holds_alternative<DirectionalCoupling>(conn) ||
           std::holds_alternative<CylindricalCoupling>(conn);
}

// ============================================================================
// ConnectionManager Implementation
// ============================================================================

ConnectionManager::ConnectionManager() = default;

void ConnectionManager::add_connection(const Connection& conn) {
    connections_.push_back(conn);
}

void ConnectionManager::clear() {
    connections_.clear();
}

bool ConnectionManager::has_spring_connections() const {
    return std::any_of(connections_.begin(), connections_.end(), is_spring_connection);
}

bool ConnectionManager::has_constraint_connections() const {
    return std::any_of(connections_.begin(), connections_.end(), is_constraint_connection);
}

std::map<std::string, size_t> ConnectionManager::get_connection_counts() const {
    std::map<std::string, size_t> counts;
    
    for (const auto& conn : connections_) {
        std::visit([&counts](const auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, SpringToGroundConnection>)
                counts["spring_to_ground"]++;
            else if constexpr (std::is_same_v<T, SpringConnection>)
                counts["spring"]++;
            else if constexpr (std::is_same_v<T, BushingConnection>)
                counts["bushing"]++;
            else if constexpr (std::is_same_v<T, RigidConnection>)
                counts[c.is_rigid ? "rbe2" : "rbe3"]++;
            else if constexpr (std::is_same_v<T, TiedConnection>)
                counts["tied"]++;
            else if constexpr (std::is_same_v<T, DirectionalCoupling>)
                counts["directional_coupling"]++;
            else if constexpr (std::is_same_v<T, CylindricalCoupling>)
                counts["cylindrical_coupling"]++;
        }, conn);
    }
    
    return counts;
}

// ============================================================================
// Find Node DOFs - Locate mesh nodes near a target point
// ============================================================================

template <int dim>
std::array<types::global_dof_index, dim> ConnectionManager::find_node_dofs(
    const Point<dim>& target,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    double tolerance) const {
    
    std::array<types::global_dof_index, dim> result;
    result.fill(numbers::invalid_dof_index);
    
    const auto& fe = dof_handler.get_fe();
    double min_dist = std::numeric_limits<double>::max();
    
    // Search all cells for closest vertex
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            Point<dim> vertex = cell->vertex(v);
            double dist = vertex.distance(target);
            
            if (dist < min_dist && dist < tolerance) {
                min_dist = dist;
                
                // Get cell DOF indices
                std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
                cell->get_dof_indices(cell_dofs);
                
                // Extract DOFs for this vertex
                // Assumes FE_Q or similar where DOFs are ordered vertex-by-vertex
                for (unsigned int d = 0; d < dim; ++d) {
                    // For FESystem<dim>(FE_Q<dim>(degree), dim), DOFs are interleaved
                    result[d] = cell_dofs[fe.component_to_system_index(d, v)];
                }
            }
        }
    }
    
    return result;
}

template <int dim>
std::vector<std::pair<Point<dim>, std::array<types::global_dof_index, dim>>>
ConnectionManager::find_boundary_nodes(
    const BoundaryTarget& target,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    std::vector<std::pair<Point<dim>, std::array<types::global_dof_index, dim>>> result;
    std::set<unsigned int> visited_vertices;
    
    const auto& fe = dof_handler.get_fe();
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            // Check if face matches target
            bool matches = false;
            if (target.type == BoundaryTarget::Type::BOUNDARY_ID) {
                matches = (cell->face(f)->boundary_id() == target.boundary_id);
            }
            // Add other target types as needed
            
            if (!matches) continue;
            
            // Get cell DOF indices
            std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
            cell->get_dof_indices(cell_dofs);
            
            // Iterate over face vertices
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                unsigned int vertex_index = cell->face(f)->vertex_index(v);
                if (visited_vertices.count(vertex_index)) continue;
                visited_vertices.insert(vertex_index);
                
                Point<dim> vertex = cell->face(f)->vertex(v);
                std::array<types::global_dof_index, dim> dofs;
                
                // Map face vertex to cell vertex
                unsigned int cell_vertex = GeometryInfo<dim>::face_to_cell_vertices(f, v);
                for (unsigned int d = 0; d < dim; ++d) {
                    dofs[d] = cell_dofs[fe.component_to_system_index(d, cell_vertex)];
                }
                
                result.emplace_back(vertex, dofs);
            }
        }
    }
    
    return result;
}

// ============================================================================
// Apply Constraints - Process constraint-type connections
// ============================================================================

template <int dim>
void ConnectionManager::apply_to_constraints(
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    for (const auto& conn : connections_) {
        std::visit([&](const auto& specific_conn) {
            using T = std::decay_t<decltype(specific_conn)>;
            
            if constexpr (std::is_same_v<T, RigidConnection>) {
                if (specific_conn.is_rigid)
                    apply_rigid_connection(specific_conn, constraints, dof_handler, mapping);
                else
                    apply_distributing_connection(specific_conn, constraints, dof_handler, mapping);
            }
            else if constexpr (std::is_same_v<T, TiedConnection>) {
                apply_tied_connection(specific_conn, constraints, dof_handler, mapping);
            }
            else if constexpr (std::is_same_v<T, DirectionalCoupling>) {
                apply_directional_coupling(specific_conn, constraints, dof_handler, mapping);
            }
            else if constexpr (std::is_same_v<T, CylindricalCoupling>) {
                apply_cylindrical_coupling(specific_conn, constraints, dof_handler, mapping);
            }
            // Spring connections are handled separately via stiffness assembly
        }, conn);
    }
}

// ============================================================================
// RBE2 Rigid Connection
// ============================================================================

template <int dim>
void ConnectionManager::apply_rigid_connection(
    const RigidConnection& conn,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    // Find master node
    auto master_dofs = find_node_dofs(conn.master_point, dof_handler, mapping,
                                       conn.node_search_tolerance);
    
    if (master_dofs[0] == numbers::invalid_dof_index) {
        throw std::runtime_error("RBE2: Master node not found at specified location");
    }
    
    // Find slave nodes
    auto slave_nodes = find_boundary_nodes(conn.slave_target, dof_handler, mapping);
    
    if (slave_nodes.empty()) {
        throw std::runtime_error("RBE2: No slave nodes found for target");
    }
    
    // Apply rigid body motion constraint
    // For 3D with translational DOFs only (no rotational DOFs in solid elements):
    // u_slave = u_master
    // 
    // Full RBE2 with rotations would be:
    // u_slave = u_master + omega × r
    // where r is the position vector from master to slave
    // This requires rotational DOFs which solid elements don't have
    
    for (const auto& [slave_point, slave_dofs] : slave_nodes) {
        // Compute offset vector (for potential rotation coupling)
        Tensor<1, dim> r;
        for (unsigned int d = 0; d < dim; ++d)
            r[d] = slave_point[d] - conn.master_point[d];
        
        // For now, apply simple translational coupling
        // u_slave_i = u_master_i
        for (unsigned int d = 0; d < dim; ++d) {
            if (!conn.coupled_dofs[d]) continue;
            if (constraints.is_constrained(slave_dofs[d])) continue;
            
            constraints.add_line(slave_dofs[d]);
            constraints.add_entry(slave_dofs[d], master_dofs[d], 1.0);
            constraints.set_inhomogeneity(slave_dofs[d], 0.0);
        }
    }
}

// ============================================================================
// RBE3 Distributing Connection
// ============================================================================

template <int dim>
void ConnectionManager::apply_distributing_connection(
    const RigidConnection& conn,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    // RBE3: Master motion = weighted average of slave motions
    // u_master = sum(w_i * u_slave_i) / sum(w_i)
    // This is the opposite of RBE2 - master is the dependent node
    
    auto master_dofs = find_node_dofs(conn.master_point, dof_handler, mapping,
                                       conn.node_search_tolerance);
    
    if (master_dofs[0] == numbers::invalid_dof_index) {
        throw std::runtime_error("RBE3: Master node not found at specified location");
    }
    
    auto slave_nodes = find_boundary_nodes(conn.slave_target, dof_handler, mapping);
    
    if (slave_nodes.empty()) {
        throw std::runtime_error("RBE3: No slave nodes found for target");
    }
    
    // Compute weights (equal weighting if not specified)
    std::vector<double> weights;
    if (conn.slave_weights.empty()) {
        weights.resize(slave_nodes.size(), 1.0 / slave_nodes.size());
    } else {
        if (conn.slave_weights.size() != slave_nodes.size()) {
            throw std::runtime_error("RBE3: Number of weights must match number of slave nodes");
        }
        weights = conn.slave_weights;
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (std::abs(total) < 1e-14) {
            throw std::runtime_error("RBE3: Sum of weights cannot be zero");
        }
        for (auto& w : weights) w /= total;
    }
    
    // For each master DOF, create constraint
    for (unsigned int d = 0; d < dim; ++d) {
        if (!conn.coupled_dofs[d]) continue;
        if (constraints.is_constrained(master_dofs[d])) continue;
        
        constraints.add_line(master_dofs[d]);
        
        for (size_t i = 0; i < slave_nodes.size(); ++i) {
            constraints.add_entry(master_dofs[d], slave_nodes[i].second[d], weights[i]);
        }
        
        constraints.set_inhomogeneity(master_dofs[d], 0.0);
    }
}

// ============================================================================
// Tied Contact Connection
// ============================================================================

template <int dim>
void ConnectionManager::apply_tied_connection(
    const TiedConnection& conn,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    // Find master and slave surface nodes
    auto master_nodes = find_boundary_nodes(conn.master_surface, dof_handler, mapping);
    auto slave_nodes = find_boundary_nodes(conn.slave_surface, dof_handler, mapping);
    
    if (master_nodes.empty()) {
        throw std::runtime_error("Tied contact: No master nodes found");
    }
    if (slave_nodes.empty()) {
        throw std::runtime_error("Tied contact: No slave nodes found");
    }
    
    // For each slave node, find closest master node and tie
    for (const auto& [slave_point, slave_dofs] : slave_nodes) {
        double min_dist = std::numeric_limits<double>::max();
        const std::array<types::global_dof_index, dim>* closest_master = nullptr;
        
        // Search for closest master node
        for (const auto& [master_point, master_dofs] : master_nodes) {
            double dist = slave_point.distance(master_point);
            if (dist < min_dist) {
                min_dist = dist;
                closest_master = &master_dofs;
            }
        }
        
        // Tie if within tolerance
        if (closest_master && min_dist <= conn.tie_tolerance) {
            for (unsigned int d = 0; d < dim; ++d) {
                if (constraints.is_constrained(slave_dofs[d])) continue;
                
                constraints.add_line(slave_dofs[d]);
                constraints.add_entry(slave_dofs[d], (*closest_master)[d], 1.0);
                constraints.set_inhomogeneity(slave_dofs[d], 0.0);
            }
        }
        else if (conn.position_tolerance_check && closest_master) {
            // Optionally warn about nodes outside tolerance
            // (could log warning here)
        }
    }
}

// ============================================================================
// Directional Coupling
// ============================================================================

template <int dim>
void ConnectionManager::apply_directional_coupling(
    const DirectionalCoupling& conn,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    auto nodes = find_boundary_nodes(conn.target, dof_handler, mapping);
    
    if (nodes.empty()) return;
    
    // Find reference node (first node or specified)
    const std::array<types::global_dof_index, dim>* ref_dofs = &nodes[0].second;
    double min_ref_dist = std::numeric_limits<double>::max();
    
    if (conn.reference_point.norm() > conn.reference_tolerance) {
        for (const auto& [pt, dofs] : nodes) {
            double dist = pt.distance(conn.reference_point);
            if (dist < min_ref_dist) {
                min_ref_dist = dist;
                ref_dofs = &dofs;
            }
        }
    }
    
    // Normalize direction
    Tensor<1, dim> dir = conn.direction;
    double dir_norm = dir.norm();
    if (dir_norm < 1e-14) return;
    dir /= dir_norm;
    
    // Constrain directional displacement to match reference
    // u · dir = u_ref · dir
    // This is: sum(dir[d] * u[d]) = sum(dir[d] * u_ref[d])
    // Or: dir[0]*u_x + dir[1]*u_y + dir[2]*u_z = dir[0]*u_ref_x + ...
    
    for (const auto& [pt, dofs] : nodes) {
        if (&dofs == ref_dofs) continue;  // Skip reference node
        
        // Choose the component with largest direction coefficient as dependent
        unsigned int dep_comp = 0;
        double max_dir = std::abs(dir[0]);
        for (unsigned int d = 1; d < dim; ++d) {
            if (std::abs(dir[d]) > max_dir) {
                max_dir = std::abs(dir[d]);
                dep_comp = d;
            }
        }
        
        if (constraints.is_constrained(dofs[dep_comp])) continue;
        
        constraints.add_line(dofs[dep_comp]);
        
        // Add contributions from other components of this node
        for (unsigned int d = 0; d < dim; ++d) {
            if (d == dep_comp) continue;
            double coef = -dir[d] / dir[dep_comp];
            constraints.add_entry(dofs[dep_comp], dofs[d], coef);
        }
        
        // Add contributions from reference node
        for (unsigned int d = 0; d < dim; ++d) {
            double coef = dir[d] / dir[dep_comp];
            constraints.add_entry(dofs[dep_comp], (*ref_dofs)[d], coef);
        }
        
        constraints.set_inhomogeneity(dofs[dep_comp], 0.0);
    }
}

// ============================================================================
// Cylindrical Coupling
// ============================================================================

template <int dim>
void ConnectionManager::apply_cylindrical_coupling(
    const CylindricalCoupling& conn,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    auto nodes = find_boundary_nodes(conn.target, dof_handler, mapping);
    if (nodes.empty()) return;
    
    // Normalize axis
    Tensor<1, dim> axis = conn.axis_direction;
    double axis_norm = axis.norm();
    if (axis_norm < 1e-14) return;
    axis /= axis_norm;
    
    // Use first node as reference
    const auto& [ref_point, ref_dofs] = nodes[0];
    
    for (size_t i = 1; i < nodes.size(); ++i) {
        const auto& [pt, dofs] = nodes[i];
        
        // Compute radial direction for this node
        Tensor<1, dim> r;
        for (unsigned int d = 0; d < dim; ++d)
            r[d] = pt[d] - conn.axis_point[d];
        
        // Remove axial component
        double axial_comp = r * axis;
        Tensor<1, dim> r_radial = r - axial_comp * axis;
        double r_mag = r_radial.norm();
        
        if (r_mag < 1e-14) continue;  // Node on axis
        
        Tensor<1, dim> radial_dir = r_radial / r_mag;
        Tensor<1, dim> circum_dir = cross_product_3d(axis, radial_dir);
        
        // Apply coupling constraints based on flags
        // This is a simplified implementation that couples displacement components
        
        if (conn.couple_axial) {
            // u · axis = u_ref · axis
            unsigned int dep_comp = 0;
            double max_axis = std::abs(axis[0]);
            for (unsigned int d = 1; d < dim; ++d) {
                if (std::abs(axis[d]) > max_axis) {
                    max_axis = std::abs(axis[d]);
                    dep_comp = d;
                }
            }
            
            if (!constraints.is_constrained(dofs[dep_comp])) {
                constraints.add_line(dofs[dep_comp]);
                
                for (unsigned int d = 0; d < dim; ++d) {
                    if (d != dep_comp) {
                        constraints.add_entry(dofs[dep_comp], dofs[d], 
                                              -axis[d] / axis[dep_comp]);
                    }
                    constraints.add_entry(dofs[dep_comp], ref_dofs[d],
                                          axis[d] / axis[dep_comp]);
                }
                constraints.set_inhomogeneity(dofs[dep_comp], 0.0);
            }
        }
        
        // Radial and circumferential coupling would require more complex
        // constraint equations relating displacements at different radii
        // This is a placeholder for the basic implementation
    }
}

// ============================================================================
// Spring Stiffness Assembly
// ============================================================================

template <int dim>
void ConnectionManager::assemble_spring_stiffness(
    SparseMatrix<double>& system_matrix,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    for (const auto& conn : connections_) {
        if (auto* stg = std::get_if<SpringToGroundConnection>(&conn)) {
            assemble_spring_to_ground(*stg, system_matrix, dof_handler, mapping);
        }
        else if (auto* sp = std::get_if<SpringConnection>(&conn)) {
            assemble_two_point_spring(*sp, system_matrix, dof_handler, mapping);
        }
        else if (auto* bush = std::get_if<BushingConnection>(&conn)) {
            assemble_bushing(*bush, system_matrix, dof_handler, mapping);
        }
    }
}

template <int dim>
void ConnectionManager::assemble_spring_to_ground(
    const SpringToGroundConnection& conn,
    SparseMatrix<double>& system_matrix,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    auto nodes = find_boundary_nodes(conn.target, dof_handler, mapping);
    
    for (const auto& [point, dofs] : nodes) {
        if (conn.use_local_coords) {
            // Transform stiffness to global coordinates
            // K_global = R * K_local * R^T
            for (unsigned int i = 0; i < dim; ++i) {
                for (unsigned int j = 0; j < dim; ++j) {
                    double k_ij = 0.0;
                    for (unsigned int k = 0; k < dim; ++k) {
                        k_ij += conn.local_to_global[i][k] * conn.stiffness[k] *
                                conn.local_to_global[j][k];
                    }
                    if (std::abs(k_ij) > 1e-14) {
                        system_matrix.add(dofs[i], dofs[j], k_ij);
                    }
                }
            }
        } else {
            // Diagonal stiffness in global coordinates
            for (unsigned int d = 0; d < dim; ++d) {
                if (conn.stiffness[d] > 0) {
                    system_matrix.add(dofs[d], dofs[d], conn.stiffness[d]);
                }
            }
        }
    }
}

template <int dim>
void ConnectionManager::assemble_two_point_spring(
    const SpringConnection& conn,
    SparseMatrix<double>& system_matrix,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    auto dofs_a = find_node_dofs(conn.point_a, dof_handler, mapping,
                                 conn.node_search_tolerance);
    auto dofs_b = find_node_dofs(conn.point_b, dof_handler, mapping,
                                 conn.node_search_tolerance);
    
    if (dofs_a[0] == numbers::invalid_dof_index ||
        dofs_b[0] == numbers::invalid_dof_index) {
        // Could not find one or both nodes
        return;
    }
    
    Tensor<1, dim> axis = conn.get_axis();
    double k = conn.axial_stiffness;
    
    // Axial spring stiffness matrix in global coordinates
    // For spring along unit vector n:
    // K_local = k * [1, -1; -1, 1]  (1D along spring axis)
    // K_global = k * n ⊗ n for each node pair
    //
    // Full 2-node stiffness matrix:
    // [K_aa  K_ab]   [ k*n⊗n   -k*n⊗n]
    // [K_ba  K_bb] = [-k*n⊗n    k*n⊗n]
    
    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
            double k_ij = k * axis[i] * axis[j];
            
            system_matrix.add(dofs_a[i], dofs_a[j], k_ij);
            system_matrix.add(dofs_a[i], dofs_b[j], -k_ij);
            system_matrix.add(dofs_b[i], dofs_a[j], -k_ij);
            system_matrix.add(dofs_b[i], dofs_b[j], k_ij);
        }
    }
    
    // Add lateral stiffness if specified
    if (conn.lateral_stiffness > 0) {
        double k_lat = conn.lateral_stiffness;
        
        // Lateral stiffness acts perpendicular to axis
        // K_lateral = k_lat * (I - n⊗n)
        for (unsigned int i = 0; i < dim; ++i) {
            for (unsigned int j = 0; j < dim; ++j) {
                double delta_ij = (i == j) ? 1.0 : 0.0;
                double k_ij = k_lat * (delta_ij - axis[i] * axis[j]);
                
                system_matrix.add(dofs_a[i], dofs_a[j], k_ij);
                system_matrix.add(dofs_a[i], dofs_b[j], -k_ij);
                system_matrix.add(dofs_b[i], dofs_a[j], -k_ij);
                system_matrix.add(dofs_b[i], dofs_b[j], k_ij);
            }
        }
    }
}

template <int dim>
void ConnectionManager::assemble_bushing(
    const BushingConnection& conn,
    SparseMatrix<double>& system_matrix,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    auto dofs_a = find_node_dofs(conn.point_a, dof_handler, mapping,
                                 conn.node_search_tolerance);
    auto dofs_b = find_node_dofs(conn.point_b, dof_handler, mapping,
                                 conn.node_search_tolerance);
    
    if (dofs_a[0] == numbers::invalid_dof_index ||
        dofs_b[0] == numbers::invalid_dof_index) {
        return;
    }
    
    // For translational DOFs only (solid elements), use translational stiffness
    // transformed by orientation matrix
    // K_global = R * K_local * R^T
    
    const auto& R = conn.orientation;
    
    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
            double k_ij = 0.0;
            for (unsigned int k = 0; k < dim; ++k) {
                k_ij += R[i][k] * conn.translational_stiffness[k] * R[j][k];
            }
            
            if (std::abs(k_ij) > 1e-14) {
                // Two-node stiffness pattern
                system_matrix.add(dofs_a[i], dofs_a[j], k_ij);
                system_matrix.add(dofs_a[i], dofs_b[j], -k_ij);
                system_matrix.add(dofs_b[i], dofs_a[j], -k_ij);
                system_matrix.add(dofs_b[i], dofs_b[j], k_ij);
            }
        }
    }
    
    // Note: Rotational stiffness would require rotational DOFs
    // or moment-curvature relationships, which are not available
    // in standard solid elements. For full bushing behavior,
    // beam or shell elements would be needed.
}

// ============================================================================
// Spring Preload Assembly
// ============================================================================

template <int dim>
void ConnectionManager::assemble_spring_preload(
    Vector<double>& system_rhs,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    for (const auto& conn : connections_) {
        if (auto* stg = std::get_if<SpringToGroundConnection>(&conn)) {
            auto nodes = find_boundary_nodes(stg->target, dof_handler, mapping);
            
            for (const auto& [point, dofs] : nodes) {
                if (stg->use_local_coords) {
                    // Transform preload to global coordinates
                    for (unsigned int i = 0; i < dim; ++i) {
                        double f_i = 0.0;
                        for (unsigned int j = 0; j < dim; ++j) {
                            f_i += stg->local_to_global[i][j] * stg->preload_force[j];
                        }
                        if (std::abs(f_i) > 1e-14) {
                            system_rhs(dofs[i]) += f_i;
                        }
                    }
                } else {
                    for (unsigned int d = 0; d < dim; ++d) {
                        if (std::abs(stg->preload_force[d]) > 1e-14) {
                            system_rhs(dofs[d]) += stg->preload_force[d];
                        }
                    }
                }
            }
        }
        else if (auto* sp = std::get_if<SpringConnection>(&conn)) {
            if (std::abs(sp->preload_force) < 1e-14) continue;
            
            auto dofs_a = find_node_dofs(sp->point_a, dof_handler, mapping,
                                         sp->node_search_tolerance);
            auto dofs_b = find_node_dofs(sp->point_b, dof_handler, mapping,
                                         sp->node_search_tolerance);
            
            if (dofs_a[0] == numbers::invalid_dof_index) continue;
            
            Tensor<1, dim> axis = sp->get_axis();
            
            // Preload acts along spring axis
            // Positive preload = tension, pulls nodes together
            for (unsigned int d = 0; d < dim; ++d) {
                double f = sp->preload_force * axis[d];
                system_rhs(dofs_a[d]) -= f;  // Pulls node a toward b
                if (dofs_b[0] != numbers::invalid_dof_index) {
                    system_rhs(dofs_b[d]) += f;  // Pulls node b toward a
                }
            }
        }
        else if (auto* bush = std::get_if<BushingConnection>(&conn)) {
            // Bushing preload
            auto dofs_a = find_node_dofs(bush->point_a, dof_handler, mapping,
                                         bush->node_search_tolerance);
            auto dofs_b = find_node_dofs(bush->point_b, dof_handler, mapping,
                                         bush->node_search_tolerance);
            
            if (dofs_a[0] == numbers::invalid_dof_index) continue;
            
            const auto& R = bush->orientation;
            
            // Transform and apply translational preload
            for (unsigned int i = 0; i < dim; ++i) {
                double f_i = 0.0;
                for (unsigned int j = 0; j < dim; ++j) {
                    f_i += R[i][j] * bush->translational_preload[j];
                }
                if (std::abs(f_i) > 1e-14) {
                    system_rhs(dofs_a[i]) -= f_i;
                    if (dofs_b[0] != numbers::invalid_dof_index) {
                        system_rhs(dofs_b[i]) += f_i;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Spring Force Calculation
// ============================================================================

template <int dim>
std::vector<typename ConnectionManager::SpringForceResult<dim>>
ConnectionManager::compute_spring_forces(
    const Vector<double>& solution,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    std::vector<SpringForceResult<dim>> results;
    
    for (const auto& conn : connections_) {
        if (auto* sp = std::get_if<SpringConnection>(&conn)) {
            auto dofs_a = find_node_dofs(sp->point_a, dof_handler, mapping,
                                         sp->node_search_tolerance);
            auto dofs_b = find_node_dofs(sp->point_b, dof_handler, mapping,
                                         sp->node_search_tolerance);
            
            if (dofs_a[0] == numbers::invalid_dof_index ||
                dofs_b[0] == numbers::invalid_dof_index) continue;
            
            // Get nodal displacements
            Tensor<1, dim> u_a, u_b;
            for (unsigned int d = 0; d < dim; ++d) {
                u_a[d] = solution(dofs_a[d]);
                u_b[d] = solution(dofs_b[d]);
            }
            
            // Compute spring elongation and force
            Tensor<1, dim> axis = sp->get_axis();
            double elongation = (u_b - u_a) * axis;  // Projection onto axis
            double axial_force = sp->axial_stiffness * elongation + sp->preload_force;
            
            SpringForceResult<dim> result;
            result.description = sp->description;
            result.location_a = sp->point_a;
            result.location_b = sp->point_b;
            result.force = axial_force * axis;
            result.axial_force = axial_force;
            result.elongation = elongation;
            
            results.push_back(result);
        }
        else if (auto* stg = std::get_if<SpringToGroundConnection>(&conn)) {
            auto nodes = find_boundary_nodes(stg->target, dof_handler, mapping);
            
            for (const auto& [pt, dofs] : nodes) {
                Tensor<1, dim> u;
                for (unsigned int d = 0; d < dim; ++d) {
                    u[d] = solution(dofs[d]);
                }
                
                // Compute spring force (F = K * u + preload)
                Tensor<1, dim> force;
                for (unsigned int d = 0; d < dim; ++d) {
                    if (stg->use_local_coords) {
                        // Transform displacement to local, compute force, transform back
                        double u_local_d = 0.0;
                        for (unsigned int j = 0; j < dim; ++j) {
                            u_local_d += stg->local_to_global[d][j] * u[j];
                        }
                        double f_local = stg->stiffness[d] * u_local_d + stg->preload_force[d];
                        for (unsigned int i = 0; i < dim; ++i) {
                            force[i] += stg->local_to_global[i][d] * f_local;
                        }
                    } else {
                        force[d] = stg->stiffness[d] * u[d] + stg->preload_force[d];
                    }
                }
                
                SpringForceResult<dim> result;
                result.description = stg->description;
                result.location_a = pt;
                result.location_b = Point<dim>();  // Ground
                result.force = force;
                result.axial_force = force.norm();
                result.elongation = u.norm();
                
                results.push_back(result);
            }
        }
    }
    
    return results;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template std::array<types::global_dof_index, 3> ConnectionManager::find_node_dofs<3>(
    const Point<3>&, const DoFHandler<3>&, const Mapping<3>&, double) const;

template std::vector<std::pair<Point<3>, std::array<types::global_dof_index, 3>>>
ConnectionManager::find_boundary_nodes<3>(
    const BoundaryTarget&, const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::apply_to_constraints<3>(
    AffineConstraints<double>&, const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::apply_rigid_connection<3>(
    const RigidConnection&, AffineConstraints<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::apply_distributing_connection<3>(
    const RigidConnection&, AffineConstraints<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::apply_tied_connection<3>(
    const TiedConnection&, AffineConstraints<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::apply_directional_coupling<3>(
    const DirectionalCoupling&, AffineConstraints<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::apply_cylindrical_coupling<3>(
    const CylindricalCoupling&, AffineConstraints<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::assemble_spring_stiffness<3>(
    SparseMatrix<double>&, const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::assemble_spring_to_ground<3>(
    const SpringToGroundConnection&, SparseMatrix<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::assemble_two_point_spring<3>(
    const SpringConnection&, SparseMatrix<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::assemble_bushing<3>(
    const BushingConnection&, SparseMatrix<double>&,
    const DoFHandler<3>&, const Mapping<3>&) const;

template void ConnectionManager::assemble_spring_preload<3>(
    Vector<double>&, const DoFHandler<3>&, const Mapping<3>&) const;

template std::vector<ConnectionManager::SpringForceResult<3>>
ConnectionManager::compute_spring_forces<3>(
    const Vector<double>&, const DoFHandler<3>&, const Mapping<3>&) const;

} // namespace FEA
