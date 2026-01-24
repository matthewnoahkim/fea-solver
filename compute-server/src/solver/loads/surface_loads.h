/**
 * @file surface_loads.h
 * @brief Surface load assembly routines
 * 
 * Implements assembly of surface loads including:
 * - Distributed surface forces (traction)
 * - Pressure loads (uniform and varying)
 * - Hydrostatic pressure
 * - Bearing loads
 */

#ifndef SURFACE_LOADS_H
#define SURFACE_LOADS_H

#include "load_base.h"
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_values.h>

namespace FEA {

using namespace dealii;

/**
 * @brief Assembler for surface force loads
 */
class SurfaceForceAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const SurfaceForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim-1>& face_quadrature);
};

/**
 * @brief Assembler for pressure loads
 */
class PressureAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const PressureLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim-1>& face_quadrature,
        const Vector<double>* current_solution = nullptr);
    
    /**
     * @brief Assemble follower load stiffness contribution
     */
    template <int dim>
    static void assemble_stiffness(
        SparseMatrix<double>& matrix,
        const PressureLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim-1>& face_quadrature,
        const Vector<double>& current_solution);
};

/**
 * @brief Assembler for hydrostatic pressure
 */
class HydrostaticPressureAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const HydrostaticPressureLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim-1>& face_quadrature);
};

/**
 * @brief Assembler for bearing loads
 */
class BearingLoadAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const BearingLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim-1>& face_quadrature);
};

} // namespace FEA

#endif // SURFACE_LOADS_H
