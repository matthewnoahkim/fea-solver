/**
 * @file body_loads.h
 * @brief Body (volume) load assembly routines
 * 
 * Implements assembly of body loads including:
 * - Gravity
 * - Linear acceleration
 * - Centrifugal loads
 */

#ifndef BODY_LOADS_H
#define BODY_LOADS_H

#include "load_base.h"
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_values.h>

namespace FEA {

using namespace dealii;

/**
 * @brief Assembler for gravity loads
 */
class GravityAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const GravityLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim>& quadrature,
        const std::map<unsigned int, Material>& materials);
};

/**
 * @brief Assembler for linear acceleration loads
 */
class LinearAccelerationAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const LinearAccelerationLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim>& quadrature,
        const std::map<unsigned int, Material>& materials);
};

/**
 * @brief Assembler for centrifugal loads
 */
class CentrifugalAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const CentrifugalLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim>& quadrature,
        const std::map<unsigned int, Material>& materials);
};

/**
 * @brief Generic body force assembler
 * 
 * Can be used with any function that provides body force as f(position, density)
 */
class GenericBodyForceAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        std::function<Tensor<1, dim>(const Point<dim>&, double)> body_force_func,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim>& quadrature,
        const std::map<unsigned int, Material>& materials,
        const std::vector<unsigned int>& material_ids = {});
};

} // namespace FEA

#endif // BODY_LOADS_H
