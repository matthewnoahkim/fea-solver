/**
 * @file surface_loads.cc
 * @brief Implementation of surface load assembly
 */

#include "surface_loads.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <cmath>

namespace FEA {

// ============================================================================
// SurfaceForceLoad Implementation
// ============================================================================

SurfaceForceLoad SurfaceForceLoad::uniform(const BoundaryTarget& target,
                                            const Tensor<1, 3>& force) {
    SurfaceForceLoad load;
    load.target = target;
    load.force_per_area = force;
    load.description = "Uniform surface force";
    return load;
}

SurfaceForceLoad SurfaceForceLoad::uniform(const BoundaryTarget& target,
                                            double fx, double fy, double fz) {
    return uniform(target, Tensor<1, 3>({fx, fy, fz}));
}

SurfaceForceLoad SurfaceForceLoad::varying(const BoundaryTarget& target,
                                            std::function<Tensor<1, 3>(const Point<3>&)> func) {
    SurfaceForceLoad load;
    load.target = target;
    load.force_function = func;
    load.description = "Varying surface force";
    return load;
}

Tensor<1, 3> SurfaceForceLoad::get_force_at(const Point<3>& p) const {
    if (force_function) {
        return force_function(p);
    }
    return force_per_area;
}

// ============================================================================
// PressureLoad Implementation
// ============================================================================

PressureLoad PressureLoad::uniform(const BoundaryTarget& target, double pressure) {
    PressureLoad load;
    load.target = target;
    load.pressure = pressure;
    load.is_follower = false;
    load.description = "Uniform pressure";
    return load;
}

PressureLoad PressureLoad::follower(const BoundaryTarget& target, double pressure) {
    PressureLoad load = uniform(target, pressure);
    load.is_follower = true;
    load.description = "Follower pressure";
    return load;
}

PressureLoad PressureLoad::varying(const BoundaryTarget& target,
                                   std::function<double(const Point<3>&)> func) {
    PressureLoad load;
    load.target = target;
    load.pressure_function = func;
    load.description = "Varying pressure";
    return load;
}

double PressureLoad::get_pressure_at(const Point<3>& p) const {
    if (pressure_function) {
        return pressure_function(p);
    }
    return pressure;
}

// ============================================================================
// HydrostaticPressureLoad Implementation
// ============================================================================

double HydrostaticPressureLoad::get_pressure_at(const Point<3>& p) const {
    // depth = (free_surface - p) · (-gravity_direction)
    Tensor<1, 3> delta;
    for (unsigned int d = 0; d < 3; ++d)
        delta[d] = free_surface_point[d] - p[d];
    
    double depth = delta * (-gravity_direction);
    return std::max(0.0, fluid_density * gravity_magnitude * depth);
}

HydrostaticPressureLoad HydrostaticPressureLoad::water(const BoundaryTarget& target,
                                                        const Point<3>& free_surface) {
    HydrostaticPressureLoad load;
    load.target = target;
    load.fluid_density = 1000.0;  // Water density [kg/m³]
    load.gravity_magnitude = 9.81;
    load.gravity_direction = Tensor<1, 3>({0, 0, -1});
    load.free_surface_point = free_surface;
    load.description = "Hydrostatic water pressure";
    return load;
}

// ============================================================================
// BearingLoad Implementation
// ============================================================================

double BearingLoad::get_pressure_at(const Point<3>& p, const Tensor<1, 3>& normal) const {
    // Vector from cylinder center to point
    Tensor<1, 3> r_vec;
    for (unsigned int d = 0; d < 3; ++d)
        r_vec[d] = p[d] - cylinder_center[d];
    
    // Project out axial component
    double axial = r_vec * cylinder_axis;
    Tensor<1, 3> radial = r_vec - axial * cylinder_axis;
    double r = radial.norm();
    
    if (r < 1e-12) return 0;
    
    // Direction from force to this point (around circumference)
    Tensor<1, 3> radial_dir = radial / r;
    
    // Force direction (perpendicular to axis)
    Tensor<1, 3> force_proj = force - (force * cylinder_axis) * cylinder_axis;
    double force_mag = force_proj.norm();
    
    if (force_mag < 1e-12) return 0;
    
    Tensor<1, 3> force_dir = force_proj / force_mag;
    
    // Angle between force direction and radial direction
    double cos_theta = force_dir * radial_dir;
    
    // Cosine distribution: p = p_max * max(0, cos(theta))
    // p_max determined from force equilibrium: F = ∫ p * cos(θ) * r * dθ * dz
    
    if (cos_theta <= 0) return 0;  // Outside contact arc
    
    // For half-cylinder contact (θ from -π/2 to π/2):
    // ∫ p_max * cos²(θ) * r dθ = p_max * r * π/2
    // So p_max = 2*F / (π * r * L) where L is contact length
    // Simplified here assuming unit length normalization
    double p_max = 4.0 * force_mag / (M_PI * cylinder_radius);
    
    return p_max * cos_theta;
}

// ============================================================================
// SurfaceForceAssembler Implementation
// ============================================================================

template <int dim>
void SurfaceForceAssembler::assemble(
    Vector<double>& rhs,
    const SurfaceForceLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim-1>& face_quadrature) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEFaceValues<dim> fe_face(mapping, fe, face_quadrature,
        update_values | update_JxW_values | update_quadrature_points);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            // Check target match
            if (load.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
                cell->face(f)->boundary_id() != load.target.boundary_id)
                continue;
            
            fe_face.reinit(cell, f);
            cell->get_dof_indices(local_dof_indices);
            local_rhs = 0;
            
            for (unsigned int q = 0; q < face_quadrature.size(); ++q) {
                const Point<dim>& q_point = fe_face.quadrature_point(q);
                const double JxW = fe_face.JxW(q);
                
                // Get force at this point
                Point<3> q_point_3d;
                for (unsigned int d = 0; d < dim && d < 3; ++d)
                    q_point_3d[d] = q_point[d];
                
                Tensor<1, 3> force_3d = load.get_force_at(q_point_3d);
                Tensor<1, dim> force;
                for (unsigned int d = 0; d < dim; ++d)
                    force[d] = force_3d[d];
                
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int comp = fe.system_to_component_index(i).first;
                    if (comp < dim) {
                        local_rhs(i) += fe_face.shape_value(i, q) * force[comp] * JxW;
                    }
                }
            }
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }
}

// ============================================================================
// PressureAssembler Implementation
// ============================================================================

template <int dim>
void PressureAssembler::assemble(
    Vector<double>& rhs,
    const PressureLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim-1>& face_quadrature,
    const Vector<double>* current_solution) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    UpdateFlags update_flags = update_values | update_JxW_values | 
                               update_quadrature_points | update_normal_vectors;
    
    if (load.is_follower && current_solution)
        update_flags |= update_gradients;
    
    FEFaceValues<dim> fe_face(mapping, fe, face_quadrature, update_flags);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            if (load.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
                cell->face(f)->boundary_id() != load.target.boundary_id)
                continue;
            
            fe_face.reinit(cell, f);
            cell->get_dof_indices(local_dof_indices);
            local_rhs = 0;
            
            for (unsigned int q = 0; q < face_quadrature.size(); ++q) {
                const Point<dim>& q_point = fe_face.quadrature_point(q);
                double JxW = fe_face.JxW(q);
                
                // Get pressure at this point
                Point<3> q_point_3d;
                for (unsigned int d = 0; d < dim && d < 3; ++d)
                    q_point_3d[d] = q_point[d];
                double p = load.get_pressure_at(q_point_3d);
                
                // Get outward normal
                Tensor<1, dim> normal = fe_face.normal_vector(q);
                
                // Traction = -p * n (negative because pressure acts inward)
                Tensor<1, dim> traction = -p * normal;
                
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int comp = fe.system_to_component_index(i).first;
                    if (comp < dim) {
                        local_rhs(i) += fe_face.shape_value(i, q) * traction[comp] * JxW;
                    }
                }
            }
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }
}

template <int dim>
void PressureAssembler::assemble_stiffness(
    SparseMatrix<double>& matrix,
    const PressureLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim-1>& face_quadrature,
    const Vector<double>& current_solution) {
    
    if (!load.is_follower) return;
    
    // Follower load stiffness: contribution from change in normal direction
    // with displacement. This is the "load stiffness" matrix.
    // K_p = -p * ∫ N^T * (∂n/∂u) * dA
    
    // For simplicity, this implementation omits the full derivation.
    // A complete implementation would compute the variation of the
    // normal vector with respect to the displacement field.
}

// ============================================================================
// HydrostaticPressureAssembler Implementation
// ============================================================================

template <int dim>
void HydrostaticPressureAssembler::assemble(
    Vector<double>& rhs,
    const HydrostaticPressureLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim-1>& face_quadrature) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEFaceValues<dim> fe_face(mapping, fe, face_quadrature,
        update_values | update_JxW_values | update_quadrature_points | update_normal_vectors);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            if (load.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
                cell->face(f)->boundary_id() != load.target.boundary_id)
                continue;
            
            fe_face.reinit(cell, f);
            cell->get_dof_indices(local_dof_indices);
            local_rhs = 0;
            
            for (unsigned int q = 0; q < face_quadrature.size(); ++q) {
                const Point<dim>& q_point = fe_face.quadrature_point(q);
                const double JxW = fe_face.JxW(q);
                const Tensor<1, dim> normal = fe_face.normal_vector(q);
                
                // Get hydrostatic pressure at this depth
                Point<3> q_point_3d;
                for (unsigned int d = 0; d < dim && d < 3; ++d)
                    q_point_3d[d] = q_point[d];
                double p = load.get_pressure_at(q_point_3d);
                
                // Traction = -p * n
                Tensor<1, dim> traction = -p * normal;
                
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int comp = fe.system_to_component_index(i).first;
                    if (comp < dim) {
                        local_rhs(i) += fe_face.shape_value(i, q) * traction[comp] * JxW;
                    }
                }
            }
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }
}

// ============================================================================
// BearingLoadAssembler Implementation
// ============================================================================

template <int dim>
void BearingLoadAssembler::assemble(
    Vector<double>& rhs,
    const BearingLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim-1>& face_quadrature) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEFaceValues<dim> fe_face(mapping, fe, face_quadrature,
        update_values | update_JxW_values | update_quadrature_points | update_normal_vectors);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            
            if (load.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
                cell->face(f)->boundary_id() != load.target.boundary_id)
                continue;
            
            fe_face.reinit(cell, f);
            cell->get_dof_indices(local_dof_indices);
            local_rhs = 0;
            
            for (unsigned int q = 0; q < face_quadrature.size(); ++q) {
                const Point<dim>& q_point = fe_face.quadrature_point(q);
                const double JxW = fe_face.JxW(q);
                const Tensor<1, dim> normal = fe_face.normal_vector(q);
                
                // Get bearing pressure at this point
                Point<3> q_point_3d;
                Tensor<1, 3> normal_3d;
                for (unsigned int d = 0; d < dim && d < 3; ++d) {
                    q_point_3d[d] = q_point[d];
                    normal_3d[d] = normal[d];
                }
                double p = load.get_pressure_at(q_point_3d, normal_3d);
                
                // Bearing load acts inward (into the hole)
                Tensor<1, dim> traction = -p * normal;
                
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int comp = fe.system_to_component_index(i).first;
                    if (comp < dim) {
                        local_rhs(i) += fe_face.shape_value(i, q) * traction[comp] * JxW;
                    }
                }
            }
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }
}

// Explicit instantiations
template void SurfaceForceAssembler::assemble<3>(
    Vector<double>&, const SurfaceForceLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<2>&);

template void SurfaceForceAssembler::assemble<2>(
    Vector<double>&, const SurfaceForceLoad&, const DoFHandler<2>&,
    const Mapping<2>&, const Quadrature<1>&);

template void PressureAssembler::assemble<3>(
    Vector<double>&, const PressureLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<2>&, const Vector<double>*);

template void PressureAssembler::assemble<2>(
    Vector<double>&, const PressureLoad&, const DoFHandler<2>&,
    const Mapping<2>&, const Quadrature<1>&, const Vector<double>*);

template void HydrostaticPressureAssembler::assemble<3>(
    Vector<double>&, const HydrostaticPressureLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<2>&);

template void BearingLoadAssembler::assemble<3>(
    Vector<double>&, const BearingLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<2>&);

} // namespace FEA
