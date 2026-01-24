/**
 * @file thermal_loads.cc
 * @brief Implementation of thermal load assembly
 */

#include "thermal_loads.h"
#include <deal.II/base/quadrature_lib.h>
#include <cmath>

namespace FEA {

// ============================================================================
// UniformThermalLoad Implementation
// ============================================================================

UniformThermalLoad UniformThermalLoad::heating(double T_ref, double T_final) {
    return UniformThermalLoad(T_ref, T_final, "Heating");
}

UniformThermalLoad UniformThermalLoad::cooling(double T_ref, double T_final) {
    return UniformThermalLoad(T_ref, T_final, "Cooling");
}

// ============================================================================
// TemperatureFieldLoad Implementation
// ============================================================================

TemperatureFieldLoad TemperatureFieldLoad::from_function(
    double T_ref,
    std::function<double(const Point<3>&)> func,
    const std::string& desc) {
    
    TemperatureFieldLoad load(T_ref);
    load.temperature_function = func;
    load.description = desc.empty() ? "Temperature field (function)" : desc;
    return load;
}

TemperatureFieldLoad TemperatureFieldLoad::from_nodal_values(
    double T_ref,
    const std::map<unsigned int, double>& temperatures,
    const std::string& desc) {
    
    TemperatureFieldLoad load(T_ref);
    load.nodal_temperatures = temperatures;
    load.description = desc.empty() ? "Temperature field (nodal)" : desc;
    return load;
}

TemperatureFieldLoad TemperatureFieldLoad::linear_gradient(
    double T_ref, double T_at_origin, const Tensor<1, 3>& gradient) {
    
    return from_function(T_ref, 
        [T_at_origin, gradient](const Point<3>& p) -> double {
            return T_at_origin + gradient[0] * p[0] + gradient[1] * p[1] + gradient[2] * p[2];
        },
        "Linear temperature gradient");
}

double TemperatureFieldLoad::get_temperature_at(const Point<3>& p) const {
    if (temperature_function) {
        return temperature_function(p);
    }
    // For nodal values, would need proper interpolation
    // Simplified: return reference temperature if no function
    return reference_temperature;
}

double TemperatureFieldLoad::get_delta_T_at(const Point<3>& p) const {
    return get_temperature_at(p) - reference_temperature;
}

// ============================================================================
// ThermalLoadAssembler Implementation
// ============================================================================

SymmetricTensor<2, 3> ThermalLoadAssembler::compute_thermal_strain(
    double delta_T,
    const Material& material) {
    
    SymmetricTensor<2, 3> thermal_strain;
    thermal_strain = 0;
    
    double alpha = material.get_thermal_expansion();
    
    // Isotropic thermal expansion: ε_th = α * ΔT * I
    for (unsigned int d = 0; d < 3; ++d) {
        thermal_strain[d][d] = alpha * delta_T;
    }
    
    // For orthotropic materials, would need directional expansion
    if (auto* ortho = std::get_if<OrthotropicElasticProperties>(&material.properties)) {
        thermal_strain[0][0] = ortho->alpha1 * delta_T;
        thermal_strain[1][1] = ortho->alpha2 * delta_T;
        thermal_strain[2][2] = ortho->alpha3 * delta_T;
        // Rotate to global coordinates
        thermal_strain = ortho->get_thermal_expansion_tensor() * delta_T;
    }
    
    return thermal_strain;
}

SymmetricTensor<2, 3> ThermalLoadAssembler::compute_thermal_stress(
    double delta_T,
    const Material& material) {
    
    SymmetricTensor<2, 3> thermal_strain = compute_thermal_strain(delta_T, material);
    SymmetricTensor<4, 3> C = material.get_elasticity_tensor();
    
    return C * thermal_strain;
}

template <int dim>
void ThermalLoadAssembler::assemble(
    Vector<double>& rhs,
    const std::vector<Load>& loads,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim>& quadrature,
    const std::map<unsigned int, Material>& materials) {
    
    for (const auto& load : loads) {
        if (auto* uniform = std::get_if<UniformThermalLoad>(&load)) {
            assemble_uniform(rhs, *uniform, dof_handler, mapping, quadrature, materials);
        }
        if (auto* field = std::get_if<TemperatureFieldLoad>(&load)) {
            assemble_field(rhs, *field, dof_handler, mapping, quadrature, materials);
        }
    }
}

template <int dim>
void ThermalLoadAssembler::assemble_uniform(
    Vector<double>& rhs,
    const UniformThermalLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim>& quadrature,
    const std::map<unsigned int, Material>& materials) {
    
    double delta_T = load.get_delta_T();
    if (std::abs(delta_T) < 1e-12) return;
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEValues<dim> fe_values(mapping, fe, quadrature,
        update_values | update_gradients | update_JxW_values);
    
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
        
        auto mat_it = materials.find(cell->material_id());
        if (mat_it == materials.end()) continue;
        
        const Material& material = mat_it->second;
        double alpha = material.get_thermal_expansion();
        
        if (std::abs(alpha) < 1e-16) continue;
        
        SymmetricTensor<4, dim> C;
        // Get elasticity tensor (simplified for dim < 3)
        auto C_3d = material.get_elasticity_tensor();
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        C[i][j][k][l] = C_3d[i][j][k][l];
        
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        local_rhs = 0;
        
        // Thermal strain (isotropic)
        SymmetricTensor<2, dim> thermal_strain;
        thermal_strain = 0;
        for (unsigned int d = 0; d < dim; ++d)
            thermal_strain[d][d] = alpha * delta_T;
        
        // Thermal stress
        SymmetricTensor<2, dim> thermal_stress = C * thermal_strain;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const double JxW = fe_values.JxW(q);
            
            // Contribution: -∫ B^T * σ_th dV
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const unsigned int comp_i = fe.system_to_component_index(i).first;
                if (comp_i >= dim) continue;
                
                // B^T * σ for this DOF (strain-displacement relation)
                double contrib = 0;
                for (unsigned int d = 0; d < dim; ++d) {
                    contrib += fe_values.shape_grad(i, q)[d] * thermal_stress[comp_i][d];
                }
                
                local_rhs(i) -= contrib * JxW;
            }
        }
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            rhs(local_dof_indices[i]) += local_rhs(i);
    }
}

template <int dim>
void ThermalLoadAssembler::assemble_field(
    Vector<double>& rhs,
    const TemperatureFieldLoad& load,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim>& quadrature,
    const std::map<unsigned int, Material>& materials) {
    
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEValues<dim> fe_values(mapping, fe, quadrature,
        update_values | update_gradients | update_JxW_values | update_quadrature_points);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        auto mat_it = materials.find(cell->material_id());
        if (mat_it == materials.end()) continue;
        
        const Material& material = mat_it->second;
        double alpha = material.get_thermal_expansion();
        
        if (std::abs(alpha) < 1e-16) continue;
        
        SymmetricTensor<4, dim> C;
        auto C_3d = material.get_elasticity_tensor();
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        C[i][j][k][l] = C_3d[i][j][k][l];
        
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        local_rhs = 0;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const Point<dim>& q_point = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);
            
            // Get temperature at this point
            Point<3> q_point_3d;
            for (unsigned int d = 0; d < dim && d < 3; ++d)
                q_point_3d[d] = q_point[d];
            
            double delta_T = load.get_delta_T_at(q_point_3d);
            
            if (std::abs(delta_T) < 1e-12) continue;
            
            // Thermal strain
            SymmetricTensor<2, dim> thermal_strain;
            thermal_strain = 0;
            for (unsigned int d = 0; d < dim; ++d)
                thermal_strain[d][d] = alpha * delta_T;
            
            // Thermal stress
            SymmetricTensor<2, dim> thermal_stress = C * thermal_strain;
            
            // Contribution: -∫ B^T * σ_th dV
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const unsigned int comp_i = fe.system_to_component_index(i).first;
                if (comp_i >= dim) continue;
                
                double contrib = 0;
                for (unsigned int d = 0; d < dim; ++d) {
                    contrib += fe_values.shape_grad(i, q)[d] * thermal_stress[comp_i][d];
                }
                
                local_rhs(i) -= contrib * JxW;
            }
        }
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            rhs(local_dof_indices[i]) += local_rhs(i);
    }
}

// ============================================================================
// TemperatureInterpolator Implementation
// ============================================================================

void TemperatureInterpolator::set_uniform(double T_ref, double T_applied) {
    reference_temperature_ = T_ref;
    uniform_temperature_ = T_applied;
    use_uniform_ = true;
    is_defined_ = true;
}

void TemperatureInterpolator::set_function(
    double T_ref, 
    std::function<double(const Point<3>&)> func) {
    
    reference_temperature_ = T_ref;
    temperature_function_ = func;
    use_uniform_ = false;
    is_defined_ = true;
}

void TemperatureInterpolator::set_nodal_values(
    double T_ref, 
    const std::map<unsigned int, double>& values) {
    
    reference_temperature_ = T_ref;
    nodal_temperatures_ = values;
    use_uniform_ = false;
    is_defined_ = true;
}

double TemperatureInterpolator::get_temperature(const Point<3>& p) const {
    if (!is_defined_) return 0;
    
    if (use_uniform_) {
        return uniform_temperature_;
    }
    
    if (temperature_function_) {
        return temperature_function_(p);
    }
    
    return reference_temperature_;
}

double TemperatureInterpolator::get_delta_T(const Point<3>& p) const {
    return get_temperature(p) - reference_temperature_;
}

// Explicit instantiations
template void ThermalLoadAssembler::assemble<3>(
    Vector<double>&, const std::vector<Load>&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<3>&, const std::map<unsigned int, Material>&);

template void ThermalLoadAssembler::assemble<2>(
    Vector<double>&, const std::vector<Load>&, const DoFHandler<2>&,
    const Mapping<2>&, const Quadrature<2>&, const std::map<unsigned int, Material>&);

template void ThermalLoadAssembler::assemble_uniform<3>(
    Vector<double>&, const UniformThermalLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<3>&, const std::map<unsigned int, Material>&);

template void ThermalLoadAssembler::assemble_field<3>(
    Vector<double>&, const TemperatureFieldLoad&, const DoFHandler<3>&,
    const Mapping<3>&, const Quadrature<3>&, const std::map<unsigned int, Material>&);

} // namespace FEA
