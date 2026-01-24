/**
 * @file point_loads.h
 * @brief Point load assembly routines
 * 
 * Implements assembly of point loads including:
 * - Concentrated forces
 * - Distributed point forces (RBF)
 * - Moments (via force couples)
 * - Remote forces
 */

#ifndef POINT_LOADS_H
#define POINT_LOADS_H

#include "load_base.h"
#include <deal.II/grid/grid_tools.h>

namespace FEA {

using namespace dealii;

/**
 * @brief Assembler for point force loads
 */
class PointForceAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const PointForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
    
private:
    template <int dim>
    static void assemble_concentrated(
        Vector<double>& rhs,
        const PointForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
    
    template <int dim>
    static void assemble_distributed(
        Vector<double>& rhs,
        const PointForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
};

/**
 * @brief Assembler for point moment loads
 */
class PointMomentAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const PointMomentLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
    
private:
    template <int dim>
    static void assemble_force_couple(
        Vector<double>& rhs,
        const PointMomentLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
};

/**
 * @brief Assembler for remote force loads
 */
class RemoteForceAssembler {
public:
    template <int dim>
    static void assemble(
        Vector<double>& rhs,
        const RemoteForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
    
private:
    template <int dim>
    static void assemble_rigid_coupling(
        Vector<double>& rhs,
        const RemoteForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
    
    template <int dim>
    static void assemble_deformable_coupling(
        Vector<double>& rhs,
        const RemoteForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping);
};

/**
 * @brief Find the nearest vertex to a point in the mesh
 */
template <int dim>
std::pair<typename DoFHandler<dim>::active_cell_iterator, unsigned int>
find_nearest_vertex(
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Point<dim>& target);

/**
 * @brief Find all vertices within a radius of a point
 */
template <int dim>
std::vector<std::pair<Point<dim>, std::vector<types::global_dof_index>>>
find_vertices_in_radius(
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Point<dim>& center,
    double radius);

/**
 * @brief RBF weight function for load distribution
 */
inline double rbf_weight(double distance, double radius, int order = 2) {
    if (distance >= radius) return 0.0;
    double r = distance / radius;
    if (order == 1)
        return 1.0 - r;  // Linear
    else if (order == 2)
        return (1.0 - r) * (1.0 - r);  // Quadratic
    else
        return std::pow(1.0 - r, order);  // Higher order
}

} // namespace FEA

#endif // POINT_LOADS_H
