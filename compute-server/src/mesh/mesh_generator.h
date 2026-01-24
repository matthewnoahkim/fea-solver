/**
 * @file mesh_generator.h
 * @brief Built-in mesh generators for simple shapes
 */

#ifndef FEA_MESH_GENERATOR_H
#define FEA_MESH_GENERATOR_H

#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>

namespace FEA {

/**
 * @brief Built-in mesh generators
 */
class MeshGenerator {
public:
    static constexpr unsigned int dim = 3;
    
    /**
     * @brief Generate a box mesh
     */
    static void box(dealii::Triangulation<dim> &triangulation,
                    const dealii::Point<dim> &p1,
                    const dealii::Point<dim> &p2,
                    const std::array<unsigned int, dim> &subdivisions);
    
    /**
     * @brief Generate a cylinder mesh
     */
    static void cylinder(dealii::Triangulation<dim> &triangulation,
                         double radius,
                         double half_length);
    
    /**
     * @brief Generate a sphere mesh
     */
    static void sphere(dealii::Triangulation<dim> &triangulation,
                       const dealii::Point<dim> &center,
                       double radius);
    
    /**
     * @brief Generate a shell (hollow sphere) mesh
     */
    static void shell(dealii::Triangulation<dim> &triangulation,
                      const dealii::Point<dim> &center,
                      double inner_radius,
                      double outer_radius);
    
    /**
     * @brief Generate a pipe (hollow cylinder) mesh
     */
    static void pipe(dealii::Triangulation<dim> &triangulation,
                     double inner_radius,
                     double outer_radius,
                     double half_length);
    
    /**
     * @brief Refine mesh globally
     */
    static void refine_global(dealii::Triangulation<dim> &triangulation,
                              unsigned int times);
};

} // namespace FEA

#endif // FEA_MESH_GENERATOR_H
