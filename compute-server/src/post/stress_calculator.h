#ifndef STRESS_CALCULATOR_H
#define STRESS_CALCULATOR_H

/**
 * @file stress_calculator.h
 * @brief Stress and strain field computation from FEA solution
 * 
 * Computes various stress measures from the displacement solution:
 * - Cauchy stress tensor components
 * - Von Mises equivalent stress
 * - Principal stresses and directions
 * - Tresca stress (maximum shear)
 * - Hydrostatic (mean) stress
 * - Strain tensor and principal strains
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_postprocessor.h>

#include "../solver/material_library.h"

#include <nlohmann/json.hpp>
#include <map>
#include <array>
#include <vector>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

/**
 * @brief Computes stress and strain fields from displacement solution
 * 
 * @tparam dim Spatial dimension (2 or 3)
 */
template <int dim>
class StressCalculator {
public:
    /**
     * @brief Construct calculator with mesh and material data
     */
    StressCalculator(const DoFHandler<dim>& dof_handler,
                     const Mapping<dim>& mapping,
                     const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Compute all stress/strain fields from displacement solution
     */
    void compute(const Vector<double>& solution);
    
    // =========================================================================
    // Field Accessors (cell-averaged values)
    // =========================================================================
    
    const Vector<double>& get_von_mises() const { return von_mises_field_; }
    const Vector<double>& get_tresca() const { return tresca_field_; }
    const Vector<double>& get_hydrostatic() const { return hydrostatic_field_; }
    const Vector<double>& get_max_shear() const { return max_shear_field_; }
    
    const std::array<Vector<double>, dim>& get_principal_stresses() const {
        return principal_stress_fields_;
    }
    
    const std::array<Vector<double>, dim>& get_principal_strains() const {
        return principal_strain_fields_;
    }
    
    // =========================================================================
    // Point Queries
    // =========================================================================
    
    /**
     * @brief Get full stress tensor at a point
     */
    SymmetricTensor<2, dim> get_stress_at_point(
        const Point<dim>& p,
        const Vector<double>& solution) const;
    
    /**
     * @brief Get full strain tensor at a point
     */
    SymmetricTensor<2, dim> get_strain_at_point(
        const Point<dim>& p,
        const Vector<double>& solution) const;
    
    /**
     * @brief Get von Mises stress at a point
     */
    double get_von_mises_at_point(
        const Point<dim>& p,
        const Vector<double>& solution) const;
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    /**
     * @brief Comprehensive stress statistics
     */
    struct StressStatistics {
        double max_von_mises;
        double min_von_mises;
        double avg_von_mises;
        double max_principal_1;         ///< Maximum σ1
        double max_principal_2;         ///< Maximum σ2
        double max_principal_3;         ///< Maximum σ3 (3D only)
        double min_principal_1;         ///< Minimum σ1
        double min_principal_2;         ///< Minimum σ2
        double min_principal_3;         ///< Minimum σ3 (3D only)
        double max_shear;
        double max_tresca;
        double max_hydrostatic;
        double min_hydrostatic;
        Point<dim> max_von_mises_location;
        Point<dim> max_principal_location;
        
        json to_json() const;
    };
    
    StressStatistics get_statistics() const;
    
    double get_max_von_mises() const { return stats_.max_von_mises; }
    Point<dim> get_max_von_mises_location() const { return stats_.max_von_mises_location; }
    
    /**
     * @brief Check if computation has been performed
     */
    bool is_computed() const { return computed_; }
    
    /**
     * @brief Get results as JSON
     */
    json to_json() const;
    
private:
    const DoFHandler<dim>& dof_handler_;
    const Mapping<dim>& mapping_;
    const std::map<unsigned int, Material>& materials_;
    
    // Cell-averaged fields
    Vector<double> von_mises_field_;
    Vector<double> tresca_field_;
    Vector<double> hydrostatic_field_;
    Vector<double> max_shear_field_;
    std::array<Vector<double>, dim> principal_stress_fields_;
    std::array<Vector<double>, dim> principal_strain_fields_;
    
    // Cached statistics
    StressStatistics stats_;
    bool computed_;
    
    // =========================================================================
    // Helper Methods
    // =========================================================================
    
    /**
     * @brief Compute strain tensor at a quadrature point
     */
    SymmetricTensor<2, dim> compute_strain_at_qpoint(
        const FEValues<dim>& fe_values,
        const std::vector<types::global_dof_index>& dof_indices,
        const Vector<double>& solution,
        unsigned int q) const;
    
    /**
     * @brief Compute stress from strain using constitutive law
     */
    SymmetricTensor<2, dim> compute_stress_from_strain(
        const SymmetricTensor<2, dim>& strain,
        unsigned int material_id) const;
    
    /**
     * @brief Compute von Mises equivalent stress
     * σ_vm = sqrt(3/2 * s:s) where s is deviatoric stress
     */
    double compute_von_mises(const SymmetricTensor<2, dim>& stress) const;
    
    /**
     * @brief Compute Tresca stress (maximum shear × 2)
     * σ_tresca = σ1 - σ3
     */
    double compute_tresca(const SymmetricTensor<2, dim>& stress) const;
    
    /**
     * @brief Compute hydrostatic (mean) stress
     * σ_h = (σ11 + σ22 + σ33) / 3
     */
    double compute_hydrostatic(const SymmetricTensor<2, dim>& stress) const;
    
    /**
     * @brief Compute principal values of a symmetric tensor
     */
    std::array<double, dim> compute_principal_values(
        const SymmetricTensor<2, dim>& tensor) const;
    
    /**
     * @brief Get elasticity tensor for a material
     */
    SymmetricTensor<4, dim> get_elasticity_tensor(unsigned int material_id) const;
};

// ============================================================================
// Data Postprocessor for VTK Output
// ============================================================================

/**
 * @brief Postprocessor to add stress tensor to VTK output
 */
template <int dim>
class StressPostprocessor : public DataPostprocessorTensor<dim> {
public:
    StressPostprocessor(const std::map<unsigned int, Material>& materials);
    
    virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& inputs,
        std::vector<Vector<double>>& computed_quantities) const override;
    
private:
    const std::map<unsigned int, Material>& materials_;
};

/**
 * @brief Postprocessor to add von Mises stress to VTK output
 */
template <int dim>
class VonMisesPostprocessor : public DataPostprocessorScalar<dim> {
public:
    VonMisesPostprocessor(const std::map<unsigned int, Material>& materials);
    
    virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& inputs,
        std::vector<Vector<double>>& computed_quantities) const override;
    
private:
    const std::map<unsigned int, Material>& materials_;
};

} // namespace FEA

#endif // STRESS_CALCULATOR_H
