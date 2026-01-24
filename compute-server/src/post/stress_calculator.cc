#include "stress_calculator.h"
#include <deal.II/lac/lapack_full_matrix.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace FEA {

// ============================================================================
// StressStatistics JSON Serialization
// ============================================================================

template <int dim>
json StressCalculator<dim>::StressStatistics::to_json() const {
    json j;
    j["von_mises"] = {
        {"max", max_von_mises},
        {"min", min_von_mises},
        {"avg", avg_von_mises},
        {"max_location", std::vector<double>{max_von_mises_location[0],
                                              max_von_mises_location[1],
                                              dim == 3 ? max_von_mises_location[2] : 0.0}}
    };
    
    j["principal"] = {
        {"sigma_1", {{"max", max_principal_1}, {"min", min_principal_1}}},
        {"sigma_2", {{"max", max_principal_2}, {"min", min_principal_2}}}
    };
    if constexpr (dim == 3) {
        j["principal"]["sigma_3"] = {{"max", max_principal_3}, {"min", min_principal_3}};
    }
    
    j["tresca"] = {{"max", max_tresca}};
    j["max_shear"] = {{"max", max_shear}};
    j["hydrostatic"] = {{"max", max_hydrostatic}, {"min", min_hydrostatic}};
    
    return j;
}

// ============================================================================
// Constructor
// ============================================================================

template <int dim>
StressCalculator<dim>::StressCalculator(
    const DoFHandler<dim>& dh,
    const Mapping<dim>& map,
    const std::map<unsigned int, Material>& mats)
    : dof_handler_(dh)
    , mapping_(map)
    , materials_(mats)
    , computed_(false)
{}

// ============================================================================
// Main Computation Method
// ============================================================================

template <int dim>
void StressCalculator<dim>::compute(const Vector<double>& solution) {
    const auto& fe = dof_handler_.get_fe();
    const unsigned int n_cells = dof_handler_.get_triangulation().n_active_cells();
    
    // Initialize fields
    von_mises_field_.reinit(n_cells);
    tresca_field_.reinit(n_cells);
    hydrostatic_field_.reinit(n_cells);
    max_shear_field_.reinit(n_cells);
    
    for (unsigned int d = 0; d < dim; ++d) {
        principal_stress_fields_[d].reinit(n_cells);
        principal_strain_fields_[d].reinit(n_cells);
    }
    
    // Setup quadrature and FE values
    QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> fe_values(mapping_, fe, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    // Initialize statistics
    stats_.max_von_mises = 0;
    stats_.min_von_mises = std::numeric_limits<double>::max();
    stats_.avg_von_mises = 0;
    stats_.max_principal_1 = std::numeric_limits<double>::lowest();
    stats_.max_principal_2 = std::numeric_limits<double>::lowest();
    stats_.max_principal_3 = std::numeric_limits<double>::lowest();
    stats_.min_principal_1 = std::numeric_limits<double>::max();
    stats_.min_principal_2 = std::numeric_limits<double>::max();
    stats_.min_principal_3 = std::numeric_limits<double>::max();
    stats_.max_shear = 0;
    stats_.max_tresca = 0;
    stats_.max_hydrostatic = std::numeric_limits<double>::lowest();
    stats_.min_hydrostatic = std::numeric_limits<double>::max();
    
    double total_volume = 0;
    double weighted_vm_sum = 0;
    
    unsigned int cell_index = 0;
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        
        double cell_volume = 0;
        double cell_vm_sum = 0;
        double cell_max_vm = 0;
        Point<dim> cell_max_vm_point;
        
        std::array<double, dim> cell_principal_sum = {};
        std::array<double, dim> cell_strain_principal_sum = {};
        double cell_hydro_sum = 0;
        double cell_tresca_max = 0;
        double cell_shear_max = 0;
        
        for (unsigned int q = 0; q < n_q_points; ++q) {
            double JxW = fe_values.JxW(q);
            cell_volume += JxW;
            
            // Compute strain tensor
            SymmetricTensor<2, dim> strain = compute_strain_at_qpoint(
                fe_values, local_dof_indices, solution, q);
            
            // Compute stress tensor
            SymmetricTensor<2, dim> stress = compute_stress_from_strain(
                strain, cell->material_id());
            
            // Compute derived quantities
            double vm = compute_von_mises(stress);
            double tresca = compute_tresca(stress);
            double hydro = compute_hydrostatic(stress);
            double max_shear = tresca / 2.0;
            
            auto stress_principals = compute_principal_values(stress);
            auto strain_principals = compute_principal_values(strain);
            
            // Accumulate for cell averaging
            cell_vm_sum += vm * JxW;
            
            if (vm > cell_max_vm) {
                cell_max_vm = vm;
                cell_max_vm_point = fe_values.quadrature_point(q);
            }
            
            for (unsigned int d = 0; d < dim; ++d) {
                cell_principal_sum[d] += stress_principals[d] * JxW;
                cell_strain_principal_sum[d] += strain_principals[d] * JxW;
            }
            
            cell_hydro_sum += hydro * JxW;
            cell_tresca_max = std::max(cell_tresca_max, tresca);
            cell_shear_max = std::max(cell_shear_max, max_shear);
            
            // Update global statistics
            if (vm > stats_.max_von_mises) {
                stats_.max_von_mises = vm;
                stats_.max_von_mises_location = fe_values.quadrature_point(q);
            }
            stats_.min_von_mises = std::min(stats_.min_von_mises, vm);
            
            // Principal stress statistics
            stats_.max_principal_1 = std::max(stats_.max_principal_1, stress_principals[0]);
            stats_.min_principal_1 = std::min(stats_.min_principal_1, stress_principals[0]);
            
            if (dim >= 2) {
                stats_.max_principal_2 = std::max(stats_.max_principal_2, stress_principals[1]);
                stats_.min_principal_2 = std::min(stats_.min_principal_2, stress_principals[1]);
            }
            
            if constexpr (dim == 3) {
                stats_.max_principal_3 = std::max(stats_.max_principal_3, stress_principals[2]);
                stats_.min_principal_3 = std::min(stats_.min_principal_3, stress_principals[2]);
            }
            
            stats_.max_shear = std::max(stats_.max_shear, max_shear);
            stats_.max_tresca = std::max(stats_.max_tresca, tresca);
            stats_.max_hydrostatic = std::max(stats_.max_hydrostatic, hydro);
            stats_.min_hydrostatic = std::min(stats_.min_hydrostatic, hydro);
        }
        
        // Store cell-averaged values
        if (cell_volume > 1e-14) {
            von_mises_field_(cell_index) = cell_vm_sum / cell_volume;
            hydrostatic_field_(cell_index) = cell_hydro_sum / cell_volume;
            
            for (unsigned int d = 0; d < dim; ++d) {
                principal_stress_fields_[d](cell_index) = cell_principal_sum[d] / cell_volume;
                principal_strain_fields_[d](cell_index) = cell_strain_principal_sum[d] / cell_volume;
            }
        }
        
        tresca_field_(cell_index) = cell_tresca_max;
        max_shear_field_(cell_index) = cell_shear_max;
        
        total_volume += cell_volume;
        weighted_vm_sum += cell_vm_sum;
        
        ++cell_index;
    }
    
    // Compute average
    if (total_volume > 1e-14) {
        stats_.avg_von_mises = weighted_vm_sum / total_volume;
    }
    
    computed_ = true;
}

// ============================================================================
// Strain Computation
// ============================================================================

template <int dim>
SymmetricTensor<2, dim> StressCalculator<dim>::compute_strain_at_qpoint(
    const FEValues<dim>& fe_values,
    const std::vector<types::global_dof_index>& dof_indices,
    const Vector<double>& solution,
    unsigned int q) const {
    
    const auto& fe = dof_handler_.get_fe();
    SymmetricTensor<2, dim> strain;
    
    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) {
        const unsigned int comp = fe.system_to_component_index(i).first;
        const double u_i = solution(dof_indices[i]);
        const Tensor<1, dim>& grad_phi = fe_values.shape_grad(i, q);
        
        // ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        for (unsigned int d = 0; d < dim; ++d) {
            strain[comp][d] += 0.5 * u_i * grad_phi[d];
            strain[d][comp] += 0.5 * u_i * grad_phi[d];
        }
    }
    
    return strain;
}

// ============================================================================
// Stress Computation
// ============================================================================

template <int dim>
SymmetricTensor<2, dim> StressCalculator<dim>::compute_stress_from_strain(
    const SymmetricTensor<2, dim>& strain,
    unsigned int material_id) const {
    
    SymmetricTensor<4, dim> C = get_elasticity_tensor(material_id);
    return C * strain;
}

template <int dim>
SymmetricTensor<4, dim> StressCalculator<dim>::get_elasticity_tensor(
    unsigned int material_id) const {
    
    auto it = materials_.find(material_id);
    if (it == materials_.end() && !materials_.empty()) {
        it = materials_.begin();
    }
    
    if (it == materials_.end()) {
        // Return default steel-like elasticity tensor
        double E = 200e9, nu = 0.3;
        double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
        double mu = E / (2 * (1 + nu));
        
        SymmetricTensor<4, dim> C;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        C[i][j][k][l] = lambda * (i == j) * (k == l) +
                                        mu * ((i == k) * (j == l) + (i == l) * (j == k));
        return C;
    }
    
    const Material& mat = it->second;
    
    if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat.properties)) {
        return iso->get_elasticity_tensor();
    }
    else if (auto* ortho = std::get_if<OrthotropicElasticProperties>(&mat.properties)) {
        return ortho->get_elasticity_tensor();
    }
    else if (auto* ep = std::get_if<ElastoplasticVonMisesProperties>(&mat.properties)) {
        return ep->get_elastic_tensor();
    }
    
    return SymmetricTensor<4, dim>();
}

// ============================================================================
// Derived Stress Quantities
// ============================================================================

template <int dim>
double StressCalculator<dim>::compute_von_mises(
    const SymmetricTensor<2, dim>& stress) const {
    
    // σ_vm = sqrt(3/2 * s_ij * s_ij) where s = deviatoric stress
    double trace = 0;
    for (unsigned int d = 0; d < dim; ++d)
        trace += stress[d][d];
    
    double mean = trace / 3.0;  // Always use 3 for 3D von Mises formula
    
    SymmetricTensor<2, dim> dev = stress;
    for (unsigned int d = 0; d < dim; ++d)
        dev[d][d] -= mean;
    
    double J2 = 0.5 * (dev * dev);
    return std::sqrt(3.0 * J2);
}

template <int dim>
double StressCalculator<dim>::compute_tresca(
    const SymmetricTensor<2, dim>& stress) const {
    
    auto principals = compute_principal_values(stress);
    // Tresca = σ_max - σ_min = σ1 - σ3
    return principals[0] - principals[dim - 1];
}

template <int dim>
double StressCalculator<dim>::compute_hydrostatic(
    const SymmetricTensor<2, dim>& stress) const {
    
    double trace = 0;
    for (unsigned int d = 0; d < dim; ++d)
        trace += stress[d][d];
    return trace / 3.0;
}

template <int dim>
std::array<double, dim> StressCalculator<dim>::compute_principal_values(
    const SymmetricTensor<2, dim>& tensor) const {
    
    auto eigen = eigenvectors(tensor);
    
    std::array<double, dim> principals;
    for (unsigned int d = 0; d < dim; ++d)
        principals[d] = eigen[d].first;
    
    // Sort in descending order (σ1 >= σ2 >= σ3)
    std::sort(principals.begin(), principals.end(), std::greater<double>());
    
    return principals;
}

// ============================================================================
// Point Queries
// ============================================================================

template <int dim>
SymmetricTensor<2, dim> StressCalculator<dim>::get_stress_at_point(
    const Point<dim>& p,
    const Vector<double>& solution) const {
    
    try {
        auto cell_and_point = GridTools::find_active_cell_around_point(
            mapping_, dof_handler_, p);
        
        auto cell = cell_and_point.first;
        auto ref_point = cell_and_point.second;
        
        if (cell == dof_handler_.end())
            return SymmetricTensor<2, dim>();
        
        Quadrature<dim> point_quadrature(ref_point);
        FEValues<dim> fe_values(mapping_, dof_handler_.get_fe(), point_quadrature,
            update_values | update_gradients);
        
        fe_values.reinit(cell);
        
        std::vector<types::global_dof_index> dof_indices(dof_handler_.get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        
        SymmetricTensor<2, dim> strain = compute_strain_at_qpoint(fe_values, dof_indices, solution, 0);
        return compute_stress_from_strain(strain, cell->material_id());
    }
    catch (...) {
        return SymmetricTensor<2, dim>();
    }
}

template <int dim>
SymmetricTensor<2, dim> StressCalculator<dim>::get_strain_at_point(
    const Point<dim>& p,
    const Vector<double>& solution) const {
    
    try {
        auto cell_and_point = GridTools::find_active_cell_around_point(
            mapping_, dof_handler_, p);
        
        auto cell = cell_and_point.first;
        auto ref_point = cell_and_point.second;
        
        if (cell == dof_handler_.end())
            return SymmetricTensor<2, dim>();
        
        Quadrature<dim> point_quadrature(ref_point);
        FEValues<dim> fe_values(mapping_, dof_handler_.get_fe(), point_quadrature,
            update_values | update_gradients);
        
        fe_values.reinit(cell);
        
        std::vector<types::global_dof_index> dof_indices(dof_handler_.get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        
        return compute_strain_at_qpoint(fe_values, dof_indices, solution, 0);
    }
    catch (...) {
        return SymmetricTensor<2, dim>();
    }
}

template <int dim>
double StressCalculator<dim>::get_von_mises_at_point(
    const Point<dim>& p,
    const Vector<double>& solution) const {
    
    return compute_von_mises(get_stress_at_point(p, solution));
}

template <int dim>
typename StressCalculator<dim>::StressStatistics
StressCalculator<dim>::get_statistics() const {
    return stats_;
}

template <int dim>
json StressCalculator<dim>::to_json() const {
    return stats_.to_json();
}

// ============================================================================
// Stress Postprocessor for VTK
// ============================================================================

template <int dim>
StressPostprocessor<dim>::StressPostprocessor(
    const std::map<unsigned int, Material>& materials)
    : DataPostprocessorTensor<dim>("stress", update_gradients)
    , materials_(materials)
{}

template <int dim>
void StressPostprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim>& inputs,
    std::vector<Vector<double>>& computed_quantities) const {
    
    const unsigned int n_q_points = inputs.solution_gradients.size();
    
    for (unsigned int q = 0; q < n_q_points; ++q) {
        // Compute strain from displacement gradients
        SymmetricTensor<2, dim> strain;
        for (unsigned int d1 = 0; d1 < dim; ++d1) {
            for (unsigned int d2 = 0; d2 < dim; ++d2) {
                strain[d1][d2] = 0.5 * (inputs.solution_gradients[q][d1][d2] +
                                        inputs.solution_gradients[q][d2][d1]);
            }
        }
        
        // Get material (use first if unknown)
        SymmetricTensor<4, dim> C;
        if (!materials_.empty()) {
            const auto& mat = materials_.begin()->second;
            if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat.properties)) {
                C = iso->get_elasticity_tensor();
            }
        }
        
        // Compute stress
        SymmetricTensor<2, dim> stress = C * strain;
        
        // Output tensor components (symmetric, so output as vector)
        unsigned int k = 0;
        for (unsigned int d1 = 0; d1 < dim; ++d1) {
            for (unsigned int d2 = d1; d2 < dim; ++d2) {
                computed_quantities[q](k++) = stress[d1][d2];
            }
        }
    }
}

// ============================================================================
// Von Mises Postprocessor for VTK
// ============================================================================

template <int dim>
VonMisesPostprocessor<dim>::VonMisesPostprocessor(
    const std::map<unsigned int, Material>& materials)
    : DataPostprocessorScalar<dim>("von_mises_stress", update_gradients)
    , materials_(materials)
{}

template <int dim>
void VonMisesPostprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim>& inputs,
    std::vector<Vector<double>>& computed_quantities) const {
    
    const unsigned int n_q_points = inputs.solution_gradients.size();
    
    for (unsigned int q = 0; q < n_q_points; ++q) {
        // Compute strain
        SymmetricTensor<2, dim> strain;
        for (unsigned int d1 = 0; d1 < dim; ++d1) {
            for (unsigned int d2 = 0; d2 < dim; ++d2) {
                strain[d1][d2] = 0.5 * (inputs.solution_gradients[q][d1][d2] +
                                        inputs.solution_gradients[q][d2][d1]);
            }
        }
        
        // Get elasticity tensor
        SymmetricTensor<4, dim> C;
        if (!materials_.empty()) {
            const auto& mat = materials_.begin()->second;
            if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat.properties)) {
                C = iso->get_elasticity_tensor();
            }
        }
        
        // Compute stress and von Mises
        SymmetricTensor<2, dim> stress = C * strain;
        
        double trace = 0;
        for (unsigned int d = 0; d < dim; ++d)
            trace += stress[d][d];
        double mean = trace / 3.0;
        
        SymmetricTensor<2, dim> dev = stress;
        for (unsigned int d = 0; d < dim; ++d)
            dev[d][d] -= mean;
        
        double J2 = 0.5 * (dev * dev);
        computed_quantities[q](0) = std::sqrt(3.0 * J2);
    }
}

// ============================================================================
// Explicit Instantiations
// ============================================================================

template class StressCalculator<3>;
template class StressCalculator<2>;

template class StressPostprocessor<3>;
template class StressPostprocessor<2>;

template class VonMisesPostprocessor<3>;
template class VonMisesPostprocessor<2>;

} // namespace FEA
