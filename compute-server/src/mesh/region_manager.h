/**
 * @file region_manager.h
 * @brief Management of mesh regions and boundary IDs
 */

#ifndef FEA_REGION_MANAGER_H
#define FEA_REGION_MANAGER_H

#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>
#include <map>
#include <string>
#include <functional>

namespace FEA {

/**
 * @brief Region information
 */
struct RegionInfo {
    std::string name;
    unsigned int id;
    double area_or_volume = 0;
    unsigned int element_count = 0;
};

/**
 * @brief Region manager for mesh
 */
class RegionManager {
public:
    static constexpr unsigned int dim = 3;
    
    using CellSelector = std::function<bool(
        const typename dealii::Triangulation<dim>::active_cell_iterator &)>;
    using FaceSelector = std::function<bool(
        const typename dealii::Triangulation<dim>::active_face_iterator &)>;
    
    /**
     * @brief Set material ID for cells matching selector
     */
    void set_material_id(dealii::Triangulation<dim> &triangulation,
                         unsigned int material_id,
                         CellSelector selector) const;
    
    /**
     * @brief Set boundary ID for faces matching selector
     */
    void set_boundary_id(dealii::Triangulation<dim> &triangulation,
                         unsigned int boundary_id,
                         FaceSelector selector) const;
    
    /**
     * @brief Set boundary ID for faces in a box region
     */
    void set_boundary_id_in_box(dealii::Triangulation<dim> &triangulation,
                                 unsigned int boundary_id,
                                 const dealii::Point<dim> &p1,
                                 const dealii::Point<dim> &p2) const;
    
    /**
     * @brief Set boundary ID for faces on a plane
     */
    void set_boundary_id_on_plane(dealii::Triangulation<dim> &triangulation,
                                   unsigned int boundary_id,
                                   const dealii::Point<dim> &point_on_plane,
                                   const dealii::Tensor<1, dim> &normal,
                                   double tolerance = 1e-10) const;
    
    /**
     * @brief Set boundary ID for faces on a cylinder surface
     */
    void set_boundary_id_on_cylinder(dealii::Triangulation<dim> &triangulation,
                                       unsigned int boundary_id,
                                       const dealii::Point<dim> &axis_point,
                                       const dealii::Tensor<1, dim> &axis_direction,
                                       double radius,
                                       double tolerance = 1e-10) const;
    
    /**
     * @brief Get all boundary IDs used in mesh
     */
    std::map<unsigned int, RegionInfo> get_boundary_info(
        const dealii::Triangulation<dim> &triangulation) const;
    
    /**
     * @brief Get all material IDs used in mesh
     */
    std::map<unsigned int, RegionInfo> get_material_info(
        const dealii::Triangulation<dim> &triangulation) const;
    
    /**
     * @brief Name a boundary region
     */
    void set_boundary_name(unsigned int boundary_id, const std::string &name);
    
    /**
     * @brief Name a material region
     */
    void set_material_name(unsigned int material_id, const std::string &name);
    
    /**
     * @brief Get boundary name
     */
    std::string get_boundary_name(unsigned int boundary_id) const;
    
    /**
     * @brief Get material name
     */
    std::string get_material_name(unsigned int material_id) const;

private:
    std::map<unsigned int, std::string> boundary_names_;
    std::map<unsigned int, std::string> material_names_;
};

} // namespace FEA

#endif // FEA_REGION_MANAGER_H
