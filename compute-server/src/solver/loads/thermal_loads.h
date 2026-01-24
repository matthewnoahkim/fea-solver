/**
 * @file thermal_loads.h
 * @brief Thermal load assembly routines
 * 
 * Implements assembly of thermal loads including:
 * - Uniform temperature change
 * - Temperature field (spatially varying)
 * - Thermal strain computation
 */

#ifndef THERMAL_LOADS_H
#define THERMAL_LOADS_H

#include "load_base.h"
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_values.h>

namespace FEA {

using namespace dealii;

/**
 * @brief Assembler for thermal loads
 * 
 * Thermal loads create initial strains ε_th = α * ΔT * I
 * This results in an equivalent RHS contribution: f = -∫ B^T * C * ε_th dV
 */
class ThermalLoadAssembler {
public:
    /**
     * @brief Assemble thermal load contributions to RHS
     */
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const std::vector<Load>& loads,  // Will extract thermal loads
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim>& quadrature,
        const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Assemble uniform thermal load
     */
    template <int dim>
    static void assemble_uniform(
        Vector<double>& rhs,
        const UniformThermalLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim>& quadrature,
        const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Assemble temperature field load
     */
    template <int dim>
    static void assemble_field(
        Vector<double>& rhs,
        const TemperatureFieldLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim>& quadrature,
        const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Compute thermal strain at a point
     */
    static SymmetricTensor<2, 3> compute_thermal_strain(
        double delta_T,
        const Material& material);
    
    /**
     * @brief Compute thermal stress at a point (for post-processing)
     */
    static SymmetricTensor<2, 3> compute_thermal_stress(
        double delta_T,
        const Material& material);
};

/**
 * @brief Temperature field interpolator
 * 
 * Handles interpolation of temperature values to quadrature points
 */
class TemperatureInterpolator {
public:
    TemperatureInterpolator() = default;
    
    /**
     * @brief Set uniform temperature
     */
    void set_uniform(double T_ref, double T_applied);
    
    /**
     * @brief Set temperature from function
     */
    void set_function(double T_ref, std::function<double(const Point<3>&)> func);
    
    /**
     * @brief Set temperature from nodal values
     */
    void set_nodal_values(double T_ref, const std::map<unsigned int, double>& values);
    
    /**
     * @brief Get temperature at a point
     */
    double get_temperature(const Point<3>& p) const;
    
    /**
     * @brief Get temperature change at a point
     */
    double get_delta_T(const Point<3>& p) const;
    
    /**
     * @brief Check if thermal loading is defined
     */
    bool is_defined() const { return is_defined_; }
    
private:
    bool is_defined_ = false;
    double reference_temperature_ = 0;
    double uniform_temperature_ = 0;
    bool use_uniform_ = true;
    std::function<double(const Point<3>&)> temperature_function_;
    std::map<unsigned int, double> nodal_temperatures_;
};

} // namespace FEA

#endif // THERMAL_LOADS_H
