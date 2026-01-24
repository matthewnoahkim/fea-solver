#include "strain_energy.h"
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <sstream>
#include <iomanip>

namespace FEA {

template <int dim>
StrainEnergyCalculator<dim>::StrainEnergyCalculator(
    const DoFHandler<dim>& dh,
    const Mapping<dim>& map,
    const std::map<unsigned int, Material>& mats)
    : dof_handler_(dh)
    , mapping_(map)
    , materials_(mats)
{}

template <int dim>
void StrainEnergyCalculator<dim>::compute(const Vector<double>& solution) {
    const auto& fe = dof_handler_.get_fe();
    const unsigned int n_cells = dof_handler_.get_triangulation().n_active_cells();
    
    sed_field_.reinit(n_cells);
    
    results_ = EnergyResults{};
    results_.total_strain_energy = 0;
    results_.max_strain_energy_density = 0;
    results_.total_volume = 0;
    
    QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> fe_values(mapping_, fe, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    
    unsigned int cell_idx = 0;
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        
        unsigned int mat_id = cell->material_id();
        SymmetricTensor<4, dim> C = get_elasticity_tensor(mat_id);
        
        double cell_energy = 0;
        double cell_volume = 0;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            // Compute strain at quadrature point
            SymmetricTensor<2, dim> strain;
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) {
                unsigned int comp = fe.system_to_component_index(i).first;
                double u_i = solution(dof_indices[i]);
                const Tensor<1, dim>& grad = fe_values.shape_grad(i, q);
                
                for (unsigned int d = 0; d < dim; ++d) {
                    strain[comp][d] += 0.5 * u_i * grad[d];
                    strain[d][comp] += 0.5 * u_i * grad[d];
                }
            }
            
            // Compute stress
            SymmetricTensor<2, dim> stress = C * strain;
            
            // Strain energy density: U = (1/2) * σ:ε
            double sed = 0.5 * (stress * strain);
            
            double JxW = fe_values.JxW(q);
            cell_energy += sed * JxW;
            cell_volume += JxW;
        }
        
        // Store cell-averaged SED
        double cell_avg_sed = (cell_volume > 1e-14) ? cell_energy / cell_volume : 0;
        sed_field_(cell_idx) = cell_avg_sed;
        
        // Track maximum
        if (cell_avg_sed > results_.max_strain_energy_density) {
            results_.max_strain_energy_density = cell_avg_sed;
            results_.max_sed_location = cell->center();
        }
        
        // Accumulate totals
        results_.total_strain_energy += cell_energy;
        results_.total_volume += cell_volume;
        
        // Track by material
        results_.energy_by_material[mat_id] += cell_energy;
        
        ++cell_idx;
    }
    
    // Compute average SED
    results_.average_sed = (results_.total_volume > 1e-14) ?
        results_.total_strain_energy / results_.total_volume : 0;
}

template <int dim>
SymmetricTensor<4, dim> StrainEnergyCalculator<dim>::get_elasticity_tensor(
    unsigned int material_id) const {
    
    auto it = materials_.find(material_id);
    if (it == materials_.end() && !materials_.empty())
        it = materials_.begin();
    
    if (it == materials_.end()) {
        // Default steel properties
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

template <int dim>
std::string StrainEnergyCalculator<dim>::get_report() const {
    std::ostringstream report;
    report << std::scientific << std::setprecision(4);
    
    report << "\n=== STRAIN ENERGY ANALYSIS ===\n\n";
    report << "Total Strain Energy:   " << results_.total_strain_energy << " J\n";
    report << "Total Volume:          " << results_.total_volume << " m³\n";
    report << "Average SED:           " << results_.average_sed << " J/m³\n";
    report << "Maximum SED:           " << results_.max_strain_energy_density << " J/m³\n";
    report << "Max SED Location:      (" << results_.max_sed_location << ")\n\n";
    
    if (!results_.energy_by_material.empty()) {
        report << "Energy by Material:\n";
        for (const auto& [id, e] : results_.energy_by_material) {
            double percent = 100.0 * e / results_.total_strain_energy;
            report << "  Material " << id << ": " << e << " J (" 
                   << std::fixed << std::setprecision(1) << percent << "%)\n";
        }
    }
    
    return report.str();
}

// Explicit instantiation
template class StrainEnergyCalculator<3>;
template class StrainEnergyCalculator<2>;

} // namespace FEA
