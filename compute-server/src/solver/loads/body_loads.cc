/**
 * @file body_loads.cc
 * @brief Implementation of body load assembly
 */

#include "body_loads.h"
#include <deal.II/base/quadrature_lib.h>
#include <algorithm>
#include <cmath>

namespace FEA {

// ============================================================================
// GravityLoad Implementation
// ============================================================================

GravityLoad GravityLoad::standard() {
    return GravityLoad(Tensor<1, 3>({0, 0, -9.81}), "Standard gravity");
}

GravityLoad GravityLoad::custom(double gx, double gy, double gz) {
    return GravityLoad(Tensor<1, 3>({gx, gy, gz}), "Custom gravity");
}

GravityLoad GravityLoad::custom(const Tensor<1, 3>& accel) {
    return GravityLoad(accel, "Custom gravity");
}

// ============================================================================
// LinearAccelerationLoad Implementation
// ============================================================================

LinearAccelerationLoad LinearAccelerationLoad::create(double ax, double ay, double az) {
    return LinearAccelerationLoad(Tensor<1, 3>({ax, ay, az}), "Linear acceleration");
}

// ============================================================================
// CentrifugalLoad Implementation
// ============================================================================

Tensor<1, 3> CentrifugalLoad::get_body_force(const Point<3>& p, double density) const {
    // Vector from axis point to the current point
    Tensor<1, 3> p_vec;
    for (unsigned int d = 0; d < 3; ++d)
        p_vec[d] = p[d] - axis_point[d];
    
    // Project out the axial component to get radial vector
    double axial_dist = p_vec * axis_direction;
    Tensor<1, 3> r_vec = p_vec - axial_dist * axis_direction;
    
    // Centrifugal force: f = ρ * ω² * r (outward)
    double omega2 = angular_velocity * angular_velocity;
    Tensor<1, 3> centrifugal = density * omega2 * r_vec;
    
    // Add tangential component from angular acceleration if present
    if (std::abs(angular_acceleration) > 1e-12) {
        // Tangential direction: t = axis × r / |r|
        double r_mag = r_vec.norm();
        if (r_mag > 1e-12) {
            Tensor<1, 3> tangent;
            tangent[0] = axis_direction[1] * r_vec[2] - axis_direction[2] * r_vec[1];
            tangent[1] = axis_direction[2] * r_vec[0] - axis_direction[0] * r_vec[2];
            tangent[2] = axis_direction[0] * r_vec[1] - axis_direction[1] * r_vec[0];
            
            // Tangential force: f_t = ρ * α * r
            centrifugal += density * angular_acceleration * tangent;
        }
    }
    
    return centrifugal;
}

CentrifugalLoad CentrifugalLoad::from_rpm(const Point<3>& axis_point,
                                          const Tensor<1, 3>& axis_direction,
                                          double rpm) {
    double omega = rpm * 2.0 * M_PI / 60.0;  // Convert RPM to rad/s
    return CentrifugalLoad(axis_point, axis_direction, omega, "Centrifugal (" + 
                          std::to_string(static_cast<int>(rpm)) + " RPM)");
}

// ============================================================================
// GravityAssembler Implementation
// ============================================================================

template <int dim>
void GravityAssembler::assemble(
    Vector<double>& rhs,
    const GravityLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim>& quadrature,
    const std::map<unsigned int, Material>& materials) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEValues<dim> fe_values(mapping, fe, quadrature,
        update_values | update_JxW_values);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    Tensor<1, dim> acceleration;
    for (unsigned int d = 0; d < dim; ++d)
        acceleration[d] = load.acceleration[d];
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        // Check material filter
        if (!load.material_ids.empty()) {
            auto it = std::find(load.material_ids.begin(), load.material_ids.end(),
                               cell->material_id());
            if (it == load.material_ids.end()) continue;
        }
        
        // Get material density
        double density = 0;
        auto mat_it = materials.find(cell->material_id());
        if (mat_it != materials.end()) {
            density = mat_it->second.get_density();
        }
        
        if (density <= 0) continue;
        
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        local_rhs = 0;
        
        // Body force = density * acceleration
        Tensor<1, dim> body_force = density * acceleration;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const double JxW = fe_values.JxW(q);
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const unsigned int comp = fe.system_to_component_index(i).first;
                if (comp < dim) {
                    local_rhs(i) += fe_values.shape_value(i, q) * body_force[comp] * JxW;
                }
            }
        }
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            rhs(local_dof_indices[i]) += local_rhs(i);
    }
}

// ============================================================================
// LinearAccelerationAssembler Implementation
// ============================================================================

template <int dim>
void LinearAccelerationAssembler::assemble(
    Vector<double>& rhs,
    const LinearAccelerationLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim>& quadrature,
    const std::map<unsigned int, Material>& materials) {
    
    // Convert to GravityLoad and use that assembler (same physics)
    GravityLoad gravity_equiv;
    gravity_equiv.acceleration = load.acceleration;
    gravity_equiv.material_ids = load.material_ids;
    gravity_equiv.description = load.description;
    
    GravityAssembler::assemble(rhs, gravity_equiv, dof_handler, mapping, 
                               quadrature, materials);
}

// ============================================================================
// CentrifugalAssembler Implementation
// ============================================================================

template <int dim>
void CentrifugalAssembler::assemble(
    Vector<double>& rhs,
    const CentrifugalLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim>& quadrature,
    const std::map<unsigned int, Material>& materials) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEValues<dim> fe_values(mapping, fe, quadrature,
        update_values | update_JxW_values | update_quadrature_points);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        // Check material filter
        if (!load.material_ids.empty()) {
            auto it = std::find(load.material_ids.begin(), load.material_ids.end(),
                               cell->material_id());
            if (it == load.material_ids.end()) continue;
        }
        
        // Get material density
        double density = 0;
        auto mat_it = materials.find(cell->material_id());
        if (mat_it != materials.end()) {
            density = mat_it->second.get_density();
        }
        
        if (density <= 0) continue;
        
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        local_rhs = 0;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const Point<dim>& q_point = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);
            
            // Convert to 3D point for body force calculation
            Point<3> q_point_3d;
            for (unsigned int d = 0; d < dim && d < 3; ++d)
                q_point_3d[d] = q_point[d];
            
            Tensor<1, 3> body_force_3d = load.get_body_force(q_point_3d, density);
            
            Tensor<1, dim> body_force;
            for (unsigned int d = 0; d < dim; ++d)
                body_force[d] = body_force_3d[d];
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const unsigned int comp = fe.system_to_component_index(i).first;
                if (comp < dim) {
                    local_rhs(i) += fe_values.shape_value(i, q) * body_force[comp] * JxW;
                }
            }
        }
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            rhs(local_dof_indices[i]) += local_rhs(i);
    }
}

// ============================================================================
// GenericBodyForceAssembler Implementation
// ============================================================================

template <int dim>
void GenericBodyForceAssembler::assemble(
    Vector<double>& rhs,
    std::function<Tensor<1, dim>(const Point<dim>&, double)> body_force_func,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim>& quadrature,
    const std::map<unsigned int, Material>& materials,
    const std::vector<unsigned int>& material_ids) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEValues<dim> fe_values(mapping, fe, quadrature,
        update_values | update_JxW_values | update_quadrature_points);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        // Check material filter
        if (!material_ids.empty()) {
            auto it = std::find(material_ids.begin(), material_ids.end(),
                               cell->material_id());
            if (it == material_ids.end()) continue;
        }
        
        // Get material density
        double density = 0;
        auto mat_it = materials.find(cell->material_id());
        if (mat_it != materials.end()) {
            density = mat_it->second.get_density();
        }
        
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        local_rhs = 0;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const Point<dim>& q_point = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);
            
            Tensor<1, dim> body_force = body_force_func(q_point, density);
            
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const unsigned int comp = fe.system_to_component_index(i).first;
                if (comp < dim) {
                    local_rhs(i) += fe_values.shape_value(i, q) * body_force[comp] * JxW;
                }
            }
        }
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            rhs(local_dof_indices[i]) += local_rhs(i);
    }
}

// Explicit instantiations
template void GravityAssembler::assemble<3>(
    Vector<double>&, const GravityLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<3>&, const std::map<unsigned int, Material>&);

template void GravityAssembler::assemble<2>(
    Vector<double>&, const GravityLoad&, const DoFHandler<2>&,
    const Mapping<2>&, const Quadrature<2>&, const std::map<unsigned int, Material>&);

template void LinearAccelerationAssembler::assemble<3>(
    Vector<double>&, const LinearAccelerationLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<3>&, const std::map<unsigned int, Material>&);

template void CentrifugalAssembler::assemble<3>(
    Vector<double>&, const CentrifugalLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<3>&, const std::map<unsigned int, Material>&);

template void GenericBodyForceAssembler::assemble<3>(
    Vector<double>&, std::function<Tensor<1, 3>(const Point<3>&, double)>,
    const DoFHandler<3>&, const Mapping<3>&, const Quadrature<3>&,
    const std::map<unsigned int, Material>&, const std::vector<unsigned int>&);

} // namespace FEA
