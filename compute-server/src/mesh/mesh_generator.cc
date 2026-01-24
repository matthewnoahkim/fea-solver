/**
 * @file mesh_generator.cc
 * @brief Implementation of mesh generators
 */

#include "mesh_generator.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

namespace FEA {

void MeshGenerator::box(dealii::Triangulation<dim> &triangulation,
                         const dealii::Point<dim> &p1,
                         const dealii::Point<dim> &p2,
                         const std::array<unsigned int, dim> &subdivisions) 
{
    std::vector<unsigned int> subs(subdivisions.begin(), subdivisions.end());
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subs, p1, p2);
}

void MeshGenerator::cylinder(dealii::Triangulation<dim> &triangulation,
                               double radius,
                               double half_length) 
{
    dealii::GridGenerator::cylinder(triangulation, radius, half_length);
}

void MeshGenerator::sphere(dealii::Triangulation<dim> &triangulation,
                             const dealii::Point<dim> &center,
                             double radius) 
{
    dealii::GridGenerator::hyper_ball(triangulation, center, radius);
}

void MeshGenerator::shell(dealii::Triangulation<dim> &triangulation,
                            const dealii::Point<dim> &center,
                            double inner_radius,
                            double outer_radius) 
{
    dealii::GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);
}

void MeshGenerator::pipe(dealii::Triangulation<dim> &triangulation,
                           double inner_radius,
                           double outer_radius,
                           double half_length) 
{
    dealii::GridGenerator::cylinder_shell(triangulation, half_length * 2,
                                           inner_radius, outer_radius);
}

void MeshGenerator::refine_global(dealii::Triangulation<dim> &triangulation,
                                    unsigned int times) 
{
    triangulation.refine_global(times);
}

} // namespace FEA
