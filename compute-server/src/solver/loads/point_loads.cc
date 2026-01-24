/**
 * @file point_loads.cc
 * @brief Implementation of point load assembly
 */

#include "point_loads.h"
#include <deal.II/grid/grid_tools.h>
#include <algorithm>
#include <cmath>
#include <set>

namespace FEA {

// ============================================================================
// PointForceLoad Implementation
// ============================================================================

PointForceLoad PointForceLoad::concentrated(const Point<3>& location,
                                            const Tensor<1, 3>& force) {
    PointForceLoad load;
    load.location = location;
    load.force = force;
    load.distribution_radius = 0.0;
    load.description = "Concentrated force";
    return load;
}

PointForceLoad PointForceLoad::concentrated(const Point<3>& location,
                                            double fx, double fy, double fz) {
    return concentrated(location, Tensor<1, 3>({fx, fy, fz}));
}

PointForceLoad PointForceLoad::distributed(const Point<3>& location,
                                           const Tensor<1, 3>& force,
                                           double radius) {
    PointForceLoad load;
    load.location = location;
    load.force = force;
    load.distribution_radius = radius;
    load.description = "Distributed point force";
    return load;
}

// ============================================================================
// PointMomentLoad Implementation
// ============================================================================

PointMomentLoad PointMomentLoad::force_couple(const Point<3>& location,
                                              const Tensor<1, 3>& moment,
                                              double radius) {
    PointMomentLoad load;
    load.location = location;
    load.moment = moment;
    load.coupling = CouplingType::FORCE_COUPLE;
    load.coupling_radius = radius;
    load.description = "Moment (force couple)";
    return load;
}

PointMomentLoad PointMomentLoad::rigid_region(const Point<3>& location,
                                              const Tensor<1, 3>& moment,
                                              double radius) {
    PointMomentLoad load;
    load.location = location;
    load.moment = moment;
    load.coupling = CouplingType::RIGID_REGION;
    load.coupling_radius = radius;
    load.description = "Moment (rigid region)";
    return load;
}

// ============================================================================
// RemoteForceLoad Implementation
// ============================================================================

RemoteForceLoad RemoteForceLoad::rigid(const Point<3>& point,
                                       const BoundaryTarget& surface,
                                       const Tensor<1, 3>& force,
                                       const Tensor<1, 3>& moment) {
    RemoteForceLoad load;
    load.application_point = point;
    load.target_surface = surface;
    load.force = force;
    load.moment = moment;
    load.coupling = CouplingType::RIGID;
    load.description = "Remote force (rigid)";
    return load;
}

RemoteForceLoad RemoteForceLoad::deformable(const Point<3>& point,
                                            const BoundaryTarget& surface,
                                            const Tensor<1, 3>& force) {
    RemoteForceLoad load;
    load.application_point = point;
    load.target_surface = surface;
    load.force = force;
    load.moment = Tensor<1, 3>();
    load.coupling = CouplingType::DEFORMABLE;
    load.description = "Remote force (deformable)";
    return load;
}

// ============================================================================
// Helper Functions
// ============================================================================

template <int dim>
std::pair<typename DoFHandler<dim>::active_cell_iterator, unsigned int>
find_nearest_vertex(
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Point<dim>& target) {
    
    typename DoFHandler<dim>::active_cell_iterator nearest_cell;
    unsigned int nearest_vertex = 0;
    double min_dist = std::numeric_limits<double>::max();
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            double dist = (cell->vertex(v) - target).norm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest_cell = cell;
                nearest_vertex = v;
            }
        }
    }
    
    return {nearest_cell, nearest_vertex};
}

template <int dim>
std::vector<std::pair<Point<dim>, std::vector<types::global_dof_index>>>
find_vertices_in_radius(
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Point<dim>& center,
    double radius) {
    
    std::vector<std::pair<Point<dim>, std::vector<types::global_dof_index>>> result;
    std::set<Point<dim>, std::function<bool(const Point<dim>&, const Point<dim>&)>> 
        seen_vertices([](const Point<dim>& a, const Point<dim>& b) {
            for (unsigned int d = 0; d < dim; ++d) {
                if (std::abs(a[d] - b[d]) > 1e-10) {
                    return a[d] < b[d];
                }
            }
            return false;
        });
    
    const auto& fe = dof_handler.get_fe();
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
        cell->get_dof_indices(cell_dofs);
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            Point<dim> vertex = cell->vertex(v);
            double dist = (vertex - center).norm();
            
            if (dist <= radius && seen_vertices.find(vertex) == seen_vertices.end()) {
                seen_vertices.insert(vertex);
                
                std::vector<types::global_dof_index> vertex_dofs;
                for (unsigned int d = 0; d < dim; ++d) {
                    // Get DOF index for this vertex and component
                    // Assumes vertex DOFs are ordered as: v0_x, v0_y, v0_z, v1_x, ...
                    vertex_dofs.push_back(cell_dofs[v * dim + d]);
                }
                
                result.emplace_back(vertex, vertex_dofs);
            }
        }
    }
    
    return result;
}

// ============================================================================
// PointForceAssembler Implementation
// ============================================================================

template <int dim>
void PointForceAssembler::assemble(
    Vector<double>& rhs,
    const PointForceLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    if (load.distribution_radius <= 0) {
        assemble_concentrated(rhs, load, dof_handler, mapping);
    } else {
        assemble_distributed(rhs, load, dof_handler, mapping);
    }
}

template <int dim>
void PointForceAssembler::assemble_concentrated(
    Vector<double>& rhs,
    const PointForceLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    // Convert 3D point to dim-D point
    Point<dim> location;
    for (unsigned int d = 0; d < dim; ++d)
        location[d] = load.location[d];
    
    // Find nearest vertex
    auto [cell, vertex_idx] = find_nearest_vertex(dof_handler, mapping, location);
    
    if (cell == dof_handler.end()) return;
    
    const auto& fe = dof_handler.get_fe();
    std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
    cell->get_dof_indices(cell_dofs);
    
    // Apply force to vertex DOFs
    for (unsigned int d = 0; d < dim; ++d) {
        types::global_dof_index dof = cell_dofs[vertex_idx * dim + d];
        rhs(dof) += load.force[d];
    }
}

template <int dim>
void PointForceAssembler::assemble_distributed(
    Vector<double>& rhs,
    const PointForceLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    Point<dim> location;
    for (unsigned int d = 0; d < dim; ++d)
        location[d] = load.location[d];
    
    // Find all vertices within radius
    auto vertices = find_vertices_in_radius(dof_handler, mapping, location, 
                                            load.distribution_radius);
    
    if (vertices.empty()) {
        // Fall back to concentrated load
        assemble_concentrated(rhs, load, dof_handler, mapping);
        return;
    }
    
    // Compute RBF weights
    double total_weight = 0;
    std::vector<double> weights;
    weights.reserve(vertices.size());
    
    for (const auto& [vertex, dofs] : vertices) {
        double dist = (vertex - location).norm();
        double w = rbf_weight(dist, load.distribution_radius, 2);
        weights.push_back(w);
        total_weight += w;
    }
    
    // Distribute force according to weights
    for (size_t i = 0; i < vertices.size(); ++i) {
        double w = weights[i] / total_weight;
        const auto& dofs = vertices[i].second;
        
        for (unsigned int d = 0; d < dim && d < dofs.size(); ++d) {
            rhs(dofs[d]) += w * load.force[d];
        }
    }
}

// ============================================================================
// PointMomentAssembler Implementation
// ============================================================================

template <int dim>
void PointMomentAssembler::assemble(
    Vector<double>& rhs,
    const PointMomentLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    if (load.coupling == PointMomentLoad::CouplingType::FORCE_COUPLE) {
        assemble_force_couple(rhs, load, dof_handler, mapping);
    } else {
        // Rigid region coupling would require constraint handling
        // Fall back to force couple for now
        assemble_force_couple(rhs, load, dof_handler, mapping);
    }
}

template <int dim>
void PointMomentAssembler::assemble_force_couple(
    Vector<double>& rhs,
    const PointMomentLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    // Convert moment to force couple
    // M = r × F  =>  F = M × r / |r|² (for perpendicular r and F)
    // We'll apply equal and opposite forces at ±radius along suitable directions
    
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
        center[d] = load.location[d];
    
    Tensor<1, 3> moment = load.moment;
    double M_mag = moment.norm();
    
    if (M_mag < 1e-12) return;
    
    Tensor<1, 3> M_dir = moment / M_mag;
    
    // Find a direction perpendicular to the moment axis
    Tensor<1, 3> r_dir;
    if (std::abs(M_dir[0]) < 0.9) {
        r_dir = Tensor<1, 3>({1, 0, 0});
    } else {
        r_dir = Tensor<1, 3>({0, 1, 0});
    }
    r_dir = r_dir - (r_dir * M_dir) * M_dir;
    r_dir /= r_dir.norm();
    
    // Force direction: F_dir = M_dir × r_dir
    Tensor<1, 3> F_dir;
    F_dir[0] = M_dir[1] * r_dir[2] - M_dir[2] * r_dir[1];
    F_dir[1] = M_dir[2] * r_dir[0] - M_dir[0] * r_dir[2];
    F_dir[2] = M_dir[0] * r_dir[1] - M_dir[1] * r_dir[0];
    
    // Force magnitude: F = M / (2 * r)
    double F_mag = M_mag / (2.0 * load.coupling_radius);
    Tensor<1, 3> force = F_mag * F_dir;
    
    // Apply +F at +r and -F at -r
    Point<3> point_plus, point_minus;
    for (unsigned int d = 0; d < 3; ++d) {
        point_plus[d] = load.location[d] + load.coupling_radius * r_dir[d];
        point_minus[d] = load.location[d] - load.coupling_radius * r_dir[d];
    }
    
    // Create point force loads and assemble
    PointForceLoad force_plus = PointForceLoad::distributed(
        point_plus, force, load.coupling_radius * 0.5);
    PointForceLoad force_minus = PointForceLoad::distributed(
        point_minus, -1.0 * force, load.coupling_radius * 0.5);
    
    assemble(rhs, force_plus, dof_handler, mapping);
    assemble(rhs, force_minus, dof_handler, mapping);
}

// ============================================================================
// RemoteForceAssembler Implementation
// ============================================================================

template <int dim>
void RemoteForceAssembler::assemble(
    Vector<double>& rhs,
    const RemoteForceLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    if (load.coupling == RemoteForceLoad::CouplingType::RIGID) {
        assemble_rigid_coupling(rhs, load, dof_handler, mapping);
    } else {
        assemble_deformable_coupling(rhs, load, dof_handler, mapping);
    }
}

template <int dim>
void RemoteForceAssembler::assemble_rigid_coupling(
    Vector<double>& rhs,
    const RemoteForceLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    // For rigid coupling (RBE2-like):
    // Forces at surface nodes are computed to be statically equivalent
    // to the applied force and moment at the application point
    
    const auto& fe = dof_handler.get_fe();
    
    // Collect all vertices on the target surface
    std::vector<std::pair<Point<dim>, std::vector<types::global_dof_index>>> surface_nodes;
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            if (load.target_surface.type == BoundaryTarget::Type::BOUNDARY_ID &&
                cell->face(f)->boundary_id() != load.target_surface.boundary_id)
                continue;
            
            std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
            cell->get_dof_indices(cell_dofs);
            
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                Point<dim> vertex = cell->face(f)->vertex(v);
                
                std::vector<types::global_dof_index> vertex_dofs;
                // Map face vertex to cell vertex index (simplified)
                unsigned int cell_vertex = v;  // Would need proper mapping
                for (unsigned int d = 0; d < dim; ++d) {
                    vertex_dofs.push_back(cell_dofs[cell_vertex * dim + d]);
                }
                
                surface_nodes.emplace_back(vertex, vertex_dofs);
            }
        }
    }
    
    if (surface_nodes.empty()) return;
    
    // Remove duplicates
    std::sort(surface_nodes.begin(), surface_nodes.end(),
        [](const auto& a, const auto& b) { return a.second[0] < b.second[0]; });
    surface_nodes.erase(std::unique(surface_nodes.begin(), surface_nodes.end(),
        [](const auto& a, const auto& b) { return a.second[0] == b.second[0]; }),
        surface_nodes.end());
    
    int n_nodes = surface_nodes.size();
    if (n_nodes == 0) return;
    
    // Compute centroid
    Point<dim> centroid;
    for (const auto& [vertex, dofs] : surface_nodes) {
        centroid += vertex;
    }
    centroid /= n_nodes;
    
    // Application point
    Point<dim> app_point;
    for (unsigned int d = 0; d < dim; ++d)
        app_point[d] = load.application_point[d];
    
    // Distribute force equally (simplified rigid distribution)
    Tensor<1, dim> force_per_node;
    for (unsigned int d = 0; d < dim; ++d)
        force_per_node[d] = load.force[d] / n_nodes;
    
    for (const auto& [vertex, dofs] : surface_nodes) {
        // Add moment contribution: F_moment = M × (node - app_point) / Σ|r|²
        Tensor<1, dim> r;
        for (unsigned int d = 0; d < dim; ++d)
            r[d] = vertex[d] - app_point[d];
        
        // F_i = F/n + (M × r_i) / Σ|r_j|²
        // Simplified: just distribute force for now
        Tensor<1, dim> node_force = force_per_node;
        
        for (unsigned int d = 0; d < dim && d < dofs.size(); ++d) {
            rhs(dofs[d]) += node_force[d];
        }
    }
}

template <int dim>
void RemoteForceAssembler::assemble_deformable_coupling(
    Vector<double>& rhs,
    const RemoteForceLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) {
    
    // For deformable coupling (RBE3-like):
    // Forces are distributed with weighting factors, typically based on
    // tributary area or distance
    
    // Similar to rigid but with distance-based weighting
    assemble_rigid_coupling(rhs, load, dof_handler, mapping);
}

// Explicit instantiations
template void PointForceAssembler::assemble<3>(
    Vector<double>&, const PointForceLoad&, const DoFHandler<3>&, const Mapping<3>&);
template void PointForceAssembler::assemble<2>(
    Vector<double>&, const PointForceLoad&, const DoFHandler<2>&, const Mapping<2>&);

template void PointMomentAssembler::assemble<3>(
    Vector<double>&, const PointMomentLoad&, const DoFHandler<3>&, const Mapping<3>&);

template void RemoteForceAssembler::assemble<3>(
    Vector<double>&, const RemoteForceLoad&, const DoFHandler<3>&, const Mapping<3>&);

template std::pair<DoFHandler<3>::active_cell_iterator, unsigned int>
find_nearest_vertex<3>(const DoFHandler<3>&, const Mapping<3>&, const Point<3>&);

template std::vector<std::pair<Point<3>, std::vector<types::global_dof_index>>>
find_vertices_in_radius<3>(const DoFHandler<3>&, const Mapping<3>&, const Point<3>&, double);

} // namespace FEA
