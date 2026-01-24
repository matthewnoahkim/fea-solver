#ifndef STRAIN_ENERGY_H
#define STRAIN_ENERGY_H

/**
 * @file strain_energy.h
 * @brief Strain energy computation for FEA results
 * 
 * Computes strain energy density and total strain energy:
 * U = (1/2) * ∫ σ:ε dV = (1/2) * ∫ ε:C:ε dV
 * 
 * Useful for:
 * - Energy-based error estimation
 * - Structural optimization
 * - Compliance minimization
 * - Identifying high-stress regions
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/mapping.h>

#include "../solver/material_library.h"

#include <nlohmann/json.hpp>
#include <map>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

/**
 * @brief Computes strain energy from FEA solution
 * 
 * @tparam dim Spatial dimension (2 or 3)
 */
template <int dim>
class StrainEnergyCalculator {
public:
    /**
     * @brief Energy computation results
     */
    struct EnergyResults {
        double total_strain_energy;         ///< Total U [J]
        double max_strain_energy_density;   ///< Max SED [J/m³]
        Point<dim> max_sed_location;        ///< Location of max SED
        std::map<unsigned int, double> energy_by_material;  ///< Energy per material
        std::map<unsigned int, double> energy_by_region;    ///< Energy per region
        double total_volume;                ///< Total model volume [m³]
        double average_sed;                 ///< Average SED [J/m³]
        
        json to_json() const {
            json j;
            j["total"] = total_strain_energy;
            j["max_density"] = max_strain_energy_density;
            j["max_density_location"] = std::vector<double>{
                max_sed_location[0], max_sed_location[1],
                dim == 3 ? max_sed_location[2] : 0.0
            };
            j["total_volume"] = total_volume;
            j["average_density"] = average_sed;
            
            json by_mat = json::object();
            for (const auto& [id, e] : energy_by_material)
                by_mat[std::to_string(id)] = e;
            j["by_material"] = by_mat;
            
            return j;
        }
    };
    
    /**
     * @brief Construct calculator
     */
    StrainEnergyCalculator(const DoFHandler<dim>& dof_handler,
                           const Mapping<dim>& mapping,
                           const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Compute strain energy from displacement solution
     */
    void compute(const Vector<double>& solution);
    
    /**
     * @brief Get results
     */
    EnergyResults get_results() const { return results_; }
    double get_total_energy() const { return results_.total_strain_energy; }
    
    /**
     * @brief Get strain energy density field (cell-averaged)
     */
    const Vector<double>& get_energy_density_field() const { return sed_field_; }
    
    /**
     * @brief Get text report
     */
    std::string get_report() const;
    
    /**
     * @brief Get JSON results
     */
    json to_json() const { return results_.to_json(); }
    
private:
    const DoFHandler<dim>& dof_handler_;
    const Mapping<dim>& mapping_;
    const std::map<unsigned int, Material>& materials_;
    
    Vector<double> sed_field_;  ///< Strain energy density per cell
    EnergyResults results_;
    
    /**
     * @brief Get elasticity tensor for material
     */
    SymmetricTensor<4, dim> get_elasticity_tensor(unsigned int material_id) const;
};

} // namespace FEA

#endif // STRAIN_ENERGY_H
