/**
 * @file region_manager.cc
 * @brief Implementation of region manager
 */

#include "region_manager.h"
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace FEA {

void RegionManager::set_material_id(dealii::Triangulation<dim> &triangulation,
                                      unsigned int material_id,
                                      CellSelector selector) const 
{
    for (auto &cell : triangulation.active_cell_iterators()) {
        if (selector(cell)) {
            cell->set_material_id(material_id);
        }
    }
}

void RegionManager::set_boundary_id(dealii::Triangulation<dim> &triangulation,
                                      unsigned int boundary_id,
                                      FaceSelector selector) const 
{
    for (auto &cell : triangulation.active_cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary() && selector(face)) {
                face->set_boundary_id(boundary_id);
            }
        }
    }
}

void RegionManager::set_boundary_id_in_box(dealii::Triangulation<dim> &triangulation,
                                             unsigned int boundary_id,
                                             const dealii::Point<dim> &p1,
                                             const dealii::Point<dim> &p2) const 
{
    dealii::Point<dim> min_pt, max_pt;
    for (unsigned int d = 0; d < dim; ++d) {
        min_pt[d] = std::min(p1[d], p2[d]);
        max_pt[d] = std::max(p1[d], p2[d]);
    }
    
    auto selector = [&](const typename dealii::Triangulation<dim>::active_face_iterator &face) {
        dealii::Point<dim> center = face->center();
        for (unsigned int d = 0; d < dim; ++d) {
            if (center[d] < min_pt[d] || center[d] > max_pt[d]) {
                return false;
            }
        }
        return true;
    };
    
    set_boundary_id(triangulation, boundary_id, selector);
}

void RegionManager::set_boundary_id_on_plane(dealii::Triangulation<dim> &triangulation,
                                                unsigned int boundary_id,
                                                const dealii::Point<dim> &point_on_plane,
                                                const dealii::Tensor<1, dim> &normal,
                                                double tolerance) const 
{
    dealii::Tensor<1, dim> n = normal / normal.norm();
    
    auto selector = [&](const typename dealii::Triangulation<dim>::active_face_iterator &face) {
        dealii::Point<dim> center = face->center();
        dealii::Tensor<1, dim> v;
        for (unsigned int d = 0; d < dim; ++d) {
            v[d] = center[d] - point_on_plane[d];
        }
        return std::abs(v * n) < tolerance;
    };
    
    set_boundary_id(triangulation, boundary_id, selector);
}

void RegionManager::set_boundary_id_on_cylinder(dealii::Triangulation<dim> &triangulation,
                                                   unsigned int boundary_id,
                                                   const dealii::Point<dim> &axis_point,
                                                   const dealii::Tensor<1, dim> &axis_direction,
                                                   double radius,
                                                   double tolerance) const 
{
    dealii::Tensor<1, dim> axis = axis_direction / axis_direction.norm();
    
    auto selector = [&](const typename dealii::Triangulation<dim>::active_face_iterator &face) {
        dealii::Point<dim> center = face->center();
        
        // Vector from axis point to face center
        dealii::Tensor<1, dim> v;
        for (unsigned int d = 0; d < dim; ++d) {
            v[d] = center[d] - axis_point[d];
        }
        
        // Remove axial component
        double axial_proj = v * axis;
        for (unsigned int d = 0; d < dim; ++d) {
            v[d] -= axial_proj * axis[d];
        }
        
        // Check radial distance
        double r = v.norm();
        return std::abs(r - radius) < tolerance;
    };
    
    set_boundary_id(triangulation, boundary_id, selector);
}

std::map<unsigned int, RegionInfo> RegionManager::get_boundary_info(
    const dealii::Triangulation<dim> &triangulation) const 
{
    std::map<unsigned int, RegionInfo> info;
    
    dealii::FE_Q<dim> fe(1);
    dealii::QGauss<dim-1> face_quadrature(2);
    dealii::FEFaceValues<dim> fe_face_values(fe, face_quadrature,
        dealii::update_JxW_values);
    
    for (const auto &cell : triangulation.active_cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary()) {
                unsigned int id = face->boundary_id();
                
                if (info.find(id) == info.end()) {
                    info[id].id = id;
                    info[id].name = get_boundary_name(id);
                }
                
                fe_face_values.reinit(cell, face);
                for (unsigned int q = 0; q < face_quadrature.size(); ++q) {
                    info[id].area_or_volume += fe_face_values.JxW(q);
                }
                info[id].element_count++;
            }
        }
    }
    
    return info;
}

std::map<unsigned int, RegionInfo> RegionManager::get_material_info(
    const dealii::Triangulation<dim> &triangulation) const 
{
    std::map<unsigned int, RegionInfo> info;
    
    dealii::FE_Q<dim> fe(1);
    dealii::QGauss<dim> quadrature(2);
    dealii::FEValues<dim> fe_values(fe, quadrature, dealii::update_JxW_values);
    
    for (const auto &cell : triangulation.active_cell_iterators()) {
        unsigned int id = cell->material_id();
        
        if (info.find(id) == info.end()) {
            info[id].id = id;
            info[id].name = get_material_name(id);
        }
        
        fe_values.reinit(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            info[id].area_or_volume += fe_values.JxW(q);
        }
        info[id].element_count++;
    }
    
    return info;
}

void RegionManager::set_boundary_name(unsigned int boundary_id, const std::string &name) {
    boundary_names_[boundary_id] = name;
}

void RegionManager::set_material_name(unsigned int material_id, const std::string &name) {
    material_names_[material_id] = name;
}

std::string RegionManager::get_boundary_name(unsigned int boundary_id) const {
    auto it = boundary_names_.find(boundary_id);
    if (it != boundary_names_.end()) {
        return it->second;
    }
    return "Boundary_" + std::to_string(boundary_id);
}

std::string RegionManager::get_material_name(unsigned int material_id) const {
    auto it = material_names_.find(material_id);
    if (it != material_names_.end()) {
        return it->second;
    }
    return "Material_" + std::to_string(material_id);
}

} // namespace FEA
