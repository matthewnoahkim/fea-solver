/**
 * @file boundary_conditions.cc
 * @brief Implementation of boundary condition system
 */

#include "boundary_conditions.h"
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace FEA {

// ============================================================================
// BoundaryTarget Implementation
// ============================================================================

BoundaryTarget BoundaryTarget::from_boundary_id(unsigned int id) {
    BoundaryTarget t;
    t.type = Type::BOUNDARY_ID;
    t.boundary_id = id;
    return t;
}

BoundaryTarget BoundaryTarget::from_material_id(unsigned int id) {
    BoundaryTarget t;
    t.type = Type::MATERIAL_ID;
    t.material_id = id;
    return t;
}

BoundaryTarget BoundaryTarget::from_point(const Point<3>& p, double tol) {
    BoundaryTarget t;
    t.type = Type::POINT;
    t.point = p;
    t.point_tolerance = tol;
    return t;
}

BoundaryTarget BoundaryTarget::from_node_set(const std::string& name) {
    BoundaryTarget t;
    t.type = Type::NODE_SET;
    t.set_name = name;
    return t;
}

BoundaryTarget BoundaryTarget::from_face_set(const std::string& name) {
    BoundaryTarget t;
    t.type = Type::FACE_SET;
    t.set_name = name;
    return t;
}

BoundaryTarget BoundaryTarget::from_box(const Point<3>& min, const Point<3>& max) {
    BoundaryTarget t;
    t.type = Type::BOX;
    t.box_min = min;
    t.box_max = max;
    return t;
}

BoundaryTarget BoundaryTarget::from_sphere(const Point<3>& center, double radius) {
    BoundaryTarget t;
    t.type = Type::SPHERE;
    t.sphere_center = center;
    t.sphere_radius = radius;
    return t;
}

BoundaryTarget BoundaryTarget::from_cylinder(const Point<3>& base, 
                                              const Tensor<1, 3>& axis,
                                              double radius, double length) {
    BoundaryTarget t;
    t.type = Type::CYLINDER;
    t.cylinder_point = base;
    t.cylinder_axis = axis / axis.norm();
    t.cylinder_radius = radius;
    t.cylinder_length = length;
    return t;
}

BoundaryTarget BoundaryTarget::from_plane(const Point<3>& point,
                                          const Tensor<1, 3>& normal, double tol) {
    BoundaryTarget t;
    t.type = Type::PLANE;
    t.plane_point = point;
    t.plane_normal = normal / normal.norm();
    t.plane_tolerance = tol;
    return t;
}

bool BoundaryTarget::contains_point(const Point<3>& p) const {
    switch (type) {
        case Type::POINT:
            return p.distance(point) <= point_tolerance;
            
        case Type::BOX:
            return p[0] >= box_min[0] && p[0] <= box_max[0] &&
                   p[1] >= box_min[1] && p[1] <= box_max[1] &&
                   p[2] >= box_min[2] && p[2] <= box_max[2];
            
        case Type::SPHERE:
            return p.distance(sphere_center) <= sphere_radius;
            
        case Type::CYLINDER: {
            Tensor<1, 3> v;
            for (unsigned int d = 0; d < 3; ++d)
                v[d] = p[d] - cylinder_point[d];
            
            double axial = v * cylinder_axis;
            if (axial < 0 || axial > cylinder_length)
                return false;
            
            Tensor<1, 3> radial = v - axial * cylinder_axis;
            return radial.norm() <= cylinder_radius;
        }
            
        case Type::PLANE: {
            Tensor<1, 3> v;
            for (unsigned int d = 0; d < 3; ++d)
                v[d] = p[d] - plane_point[d];
            
            double dist = std::abs(v * plane_normal);
            return dist <= plane_tolerance;
        }
            
        default:
            return false;
    }
}

std::string BoundaryTarget::to_string() const {
    std::ostringstream oss;
    switch (type) {
        case Type::BOUNDARY_ID:
            oss << "boundary_id=" << boundary_id;
            break;
        case Type::MATERIAL_ID:
            oss << "material_id=" << material_id;
            break;
        case Type::POINT:
            oss << "point(" << point << ")";
            break;
        case Type::NODE_SET:
            oss << "node_set=" << set_name;
            break;
        case Type::FACE_SET:
            oss << "face_set=" << set_name;
            break;
        case Type::BOX:
            oss << "box(" << box_min << " to " << box_max << ")";
            break;
        case Type::SPHERE:
            oss << "sphere(center=" << sphere_center << ", r=" << sphere_radius << ")";
            break;
        case Type::CYLINDER:
            oss << "cylinder(r=" << cylinder_radius << ")";
            break;
        case Type::PLANE:
            oss << "plane(n=" << plane_normal << ")";
            break;
    }
    return oss.str();
}

// ============================================================================
// CoordinateSystem Implementation
// ============================================================================

CoordinateSystem CoordinateSystem::global() {
    CoordinateSystem cs;
    cs.type = Type::GLOBAL;
    return cs;
}

CoordinateSystem CoordinateSystem::cylindrical(const Point<3>& origin,
                                                const Tensor<1, 3>& axis) {
    CoordinateSystem cs;
    cs.type = Type::CYLINDRICAL;
    cs.axis_origin = origin;
    cs.axis_direction = axis / axis.norm();
    return cs;
}

CoordinateSystem CoordinateSystem::spherical(const Point<3>& center) {
    CoordinateSystem cs;
    cs.type = Type::SPHERICAL;
    cs.center = center;
    return cs;
}

CoordinateSystem CoordinateSystem::local(const Tensor<2, 3>& R, const Point<3>& origin) {
    CoordinateSystem cs;
    cs.type = Type::LOCAL;
    cs.rotation_matrix = R;
    cs.origin = origin;
    return cs;
}

Tensor<2, 3> CoordinateSystem::get_transformation(const Point<3>& at_point) const {
    Tensor<2, 3> T;
    T = 0;
    
    switch (type) {
        case Type::GLOBAL:
            // Identity transformation
            for (unsigned int i = 0; i < 3; ++i)
                T[i][i] = 1.0;
            break;
            
        case Type::CYLINDRICAL: {
            // Build local basis: r, θ, z
            Tensor<1, 3> z_dir = axis_direction;
            Tensor<1, 3> p_vec;
            for (unsigned int d = 0; d < 3; ++d)
                p_vec[d] = at_point[d] - axis_origin[d];
            
            // Radial direction = p - (p·z)z
            Tensor<1, 3> r_vec = p_vec - (p_vec * z_dir) * z_dir;
            double r_mag = r_vec.norm();
            
            Tensor<1, 3> r_dir, theta_dir;
            if (r_mag > 1e-12) {
                r_dir = r_vec / r_mag;
                // θ = z × r
                theta_dir[0] = z_dir[1] * r_dir[2] - z_dir[2] * r_dir[1];
                theta_dir[1] = z_dir[2] * r_dir[0] - z_dir[0] * r_dir[2];
                theta_dir[2] = z_dir[0] * r_dir[1] - z_dir[1] * r_dir[0];
            } else {
                // On axis - pick arbitrary r direction
                Tensor<1, 3> temp;
                if (std::abs(z_dir[0]) < 0.9) {
                    temp[0] = 1; temp[1] = 0; temp[2] = 0;
                } else {
                    temp[0] = 0; temp[1] = 1; temp[2] = 0;
                }
                r_dir = temp - (temp * z_dir) * z_dir;
                r_dir /= r_dir.norm();
                theta_dir[0] = z_dir[1] * r_dir[2] - z_dir[2] * r_dir[1];
                theta_dir[1] = z_dir[2] * r_dir[0] - z_dir[0] * r_dir[2];
                theta_dir[2] = z_dir[0] * r_dir[1] - z_dir[1] * r_dir[0];
            }
            
            // Columns are local basis vectors: [r, θ, z]
            for (unsigned int d = 0; d < 3; ++d) {
                T[d][0] = r_dir[d];
                T[d][1] = theta_dir[d];
                T[d][2] = z_dir[d];
            }
            break;
        }
            
        case Type::SPHERICAL: {
            // Build local basis: r, θ, φ
            Tensor<1, 3> p_vec;
            for (unsigned int d = 0; d < 3; ++d)
                p_vec[d] = at_point[d] - center[d];
            
            double r_mag = p_vec.norm();
            if (r_mag < 1e-12) {
                // At center - use identity
                for (unsigned int i = 0; i < 3; ++i)
                    T[i][i] = 1.0;
            } else {
                Tensor<1, 3> r_dir = p_vec / r_mag;
                
                // φ direction (longitude) - perpendicular to z and r
                Tensor<1, 3> z_axis({0, 0, 1});
                Tensor<1, 3> phi_dir;
                phi_dir[0] = z_axis[1] * r_dir[2] - z_axis[2] * r_dir[1];
                phi_dir[1] = z_axis[2] * r_dir[0] - z_axis[0] * r_dir[2];
                phi_dir[2] = z_axis[0] * r_dir[1] - z_axis[1] * r_dir[0];
                double phi_mag = phi_dir.norm();
                
                if (phi_mag > 1e-12) {
                    phi_dir /= phi_mag;
                } else {
                    // At poles
                    phi_dir[0] = 1; phi_dir[1] = 0; phi_dir[2] = 0;
                }
                
                // θ direction = φ × r
                Tensor<1, 3> theta_dir;
                theta_dir[0] = phi_dir[1] * r_dir[2] - phi_dir[2] * r_dir[1];
                theta_dir[1] = phi_dir[2] * r_dir[0] - phi_dir[0] * r_dir[2];
                theta_dir[2] = phi_dir[0] * r_dir[1] - phi_dir[1] * r_dir[0];
                
                for (unsigned int d = 0; d < 3; ++d) {
                    T[d][0] = r_dir[d];
                    T[d][1] = theta_dir[d];
                    T[d][2] = phi_dir[d];
                }
            }
            break;
        }
            
        case Type::LOCAL:
            T = rotation_matrix;
            break;
    }
    
    return T;
}

Tensor<1, 3> CoordinateSystem::to_global(const Tensor<1, 3>& local_disp,
                                          const Point<3>& at_point) const {
    return get_transformation(at_point) * local_disp;
}

Tensor<1, 3> CoordinateSystem::to_local(const Tensor<1, 3>& global_disp,
                                         const Point<3>& at_point) const {
    return transpose(get_transformation(at_point)) * global_disp;
}

// ============================================================================
// Boundary Condition Type Implementations
// ============================================================================

DisplacementBC DisplacementBC::fixed(const BoundaryTarget& target) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {0.0, 0.0, 0.0};
    bc.description = "Fixed support";
    return bc;
}

DisplacementBC DisplacementBC::x_only(const BoundaryTarget& target, double ux) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {ux, std::nullopt, std::nullopt};
    bc.description = "X-displacement constraint";
    return bc;
}

DisplacementBC DisplacementBC::y_only(const BoundaryTarget& target, double uy) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {std::nullopt, uy, std::nullopt};
    bc.description = "Y-displacement constraint";
    return bc;
}

DisplacementBC DisplacementBC::z_only(const BoundaryTarget& target, double uz) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {std::nullopt, std::nullopt, uz};
    bc.description = "Z-displacement constraint";
    return bc;
}

DisplacementBC DisplacementBC::xy_only(const BoundaryTarget& target, double ux, double uy) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {ux, uy, std::nullopt};
    bc.description = "XY-displacement constraint";
    return bc;
}

DisplacementBC DisplacementBC::xz_only(const BoundaryTarget& target, double ux, double uz) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {ux, std::nullopt, uz};
    bc.description = "XZ-displacement constraint";
    return bc;
}

DisplacementBC DisplacementBC::yz_only(const BoundaryTarget& target, double uy, double uz) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {std::nullopt, uy, uz};
    bc.description = "YZ-displacement constraint";
    return bc;
}

DisplacementBC DisplacementBC::full(const BoundaryTarget& target,
                                     double ux, double uy, double uz) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {ux, uy, uz};
    bc.description = "Full displacement constraint";
    return bc;
}

DisplacementBC DisplacementBC::radial_only(const BoundaryTarget& target, double ur,
                                           const Point<3>& axis_origin, 
                                           const Tensor<1, 3>& axis) {
    DisplacementBC bc;
    bc.target = target;
    bc.values = {ur, std::nullopt, std::nullopt};  // r-component only
    bc.coord_system = CoordinateSystem::cylindrical(axis_origin, axis);
    bc.description = "Radial displacement constraint";
    return bc;
}

int DisplacementBC::num_constrained() const {
    int count = 0;
    for (const auto& v : values)
        if (v.has_value()) ++count;
    return count;
}

SymmetryBC SymmetryBC::x_plane(const BoundaryTarget& target) {
    return SymmetryBC(target, Tensor<1, 3>({1, 0, 0}), "X-symmetry plane");
}

SymmetryBC SymmetryBC::y_plane(const BoundaryTarget& target) {
    return SymmetryBC(target, Tensor<1, 3>({0, 1, 0}), "Y-symmetry plane");
}

SymmetryBC SymmetryBC::z_plane(const BoundaryTarget& target) {
    return SymmetryBC(target, Tensor<1, 3>({0, 0, 1}), "Z-symmetry plane");
}

ElasticSupportBC ElasticSupportBC::uniform(const BoundaryTarget& target, double k) {
    return ElasticSupportBC(target, Tensor<1, 3>({k, k, k}), "Uniform elastic support");
}

ElasticSupportBC ElasticSupportBC::normal_only(const BoundaryTarget& target, double k_n) {
    ElasticSupportBC bc;
    bc.target = target;
    bc.use_local_directions = true;
    bc.normal_stiffness = k_n;
    bc.tangential_stiffness = 0;
    bc.description = "Normal-only elastic support";
    return bc;
}

ElasticSupportBC ElasticSupportBC::tangential_only(const BoundaryTarget& target, double k_t) {
    ElasticSupportBC bc;
    bc.target = target;
    bc.use_local_directions = true;
    bc.normal_stiffness = 0;
    bc.tangential_stiffness = k_t;
    bc.description = "Tangential-only elastic support";
    return bc;
}

ElasticSupportBC ElasticSupportBC::anisotropic(const BoundaryTarget& target,
                                               double kx, double ky, double kz) {
    return ElasticSupportBC(target, Tensor<1, 3>({kx, ky, kz}), "Anisotropic elastic support");
}

bool FrictionlessSupportBC::is_contact_active(const Tensor<1, 3>& displacement) const {
    double gap = initial_gap - displacement * surface_normal;
    return gap < 0;  // Penetration
}

Tensor<1, 3> FrictionlessSupportBC::get_contact_force(const Tensor<1, 3>& displacement) const {
    double gap = initial_gap - displacement * surface_normal;
    if (gap >= 0) {
        return Tensor<1, 3>();  // No contact
    }
    // Penalty force: F = -k * gap * n (pushes back)
    return -penalty_stiffness * gap * surface_normal;
}

// ============================================================================
// Variant Helpers
// ============================================================================

const BoundaryTarget& get_target(const BoundaryCondition& bc) {
    return std::visit([](const auto& b) -> const BoundaryTarget& {
        if constexpr (std::is_same_v<std::decay_t<decltype(b)>, CyclicSymmetryBC>)
            return b.master_target;
        else
            return b.target;
    }, bc);
}

const std::string& get_description(const BoundaryCondition& bc) {
    return std::visit([](const auto& b) -> const std::string& {
        return b.description;
    }, bc);
}

bool is_dirichlet_bc(const BoundaryCondition& bc) {
    return std::holds_alternative<FixedBC>(bc) ||
           std::holds_alternative<DisplacementBC>(bc) ||
           std::holds_alternative<SymmetryBC>(bc);
}

bool is_neumann_bc(const BoundaryCondition& bc) {
    return std::holds_alternative<ElasticSupportBC>(bc) ||
           std::holds_alternative<FrictionlessSupportBC>(bc);
}

bool is_nonlinear_bc(const BoundaryCondition& bc) {
    return std::holds_alternative<FrictionlessSupportBC>(bc);
}

bool requires_coupling(const BoundaryCondition& bc) {
    return std::holds_alternative<CyclicSymmetryBC>(bc);
}

// ============================================================================
// BoundaryConditionManager Implementation
// ============================================================================

BoundaryConditionManager::BoundaryConditionManager() = default;

void BoundaryConditionManager::add_condition(const BoundaryCondition& bc) {
    conditions_.push_back(bc);
}

void BoundaryConditionManager::add_conditions(const std::vector<BoundaryCondition>& bcs) {
    conditions_.insert(conditions_.end(), bcs.begin(), bcs.end());
}

void BoundaryConditionManager::remove_condition(size_t index) {
    if (index < conditions_.size()) {
        conditions_.erase(conditions_.begin() + index);
    }
}

void BoundaryConditionManager::clear() {
    conditions_.clear();
    active_contact_dofs_.clear();
}

bool BoundaryConditionManager::has_contact_conditions() const {
    return std::any_of(conditions_.begin(), conditions_.end(),
        [](const BoundaryCondition& bc) {
            return std::holds_alternative<FrictionlessSupportBC>(bc);
        });
}

bool BoundaryConditionManager::has_elastic_support_conditions() const {
    return std::any_of(conditions_.begin(), conditions_.end(),
        [](const BoundaryCondition& bc) {
            return std::holds_alternative<ElasticSupportBC>(bc);
        });
}

bool BoundaryConditionManager::has_cyclic_symmetry() const {
    return std::any_of(conditions_.begin(), conditions_.end(),
        [](const BoundaryCondition& bc) {
            return std::holds_alternative<CyclicSymmetryBC>(bc);
        });
}

bool BoundaryConditionManager::has_nonlinear_conditions() const {
    return std::any_of(conditions_.begin(), conditions_.end(), is_nonlinear_bc);
}

std::set<types::boundary_id> BoundaryConditionManager::get_dirichlet_boundary_ids() const {
    std::set<types::boundary_id> ids;
    for (const auto& bc : conditions_) {
        if (is_dirichlet_bc(bc)) {
            const auto& target = get_target(bc);
            if (target.type == BoundaryTarget::Type::BOUNDARY_ID)
                ids.insert(target.boundary_id);
        }
    }
    return ids;
}

std::set<types::boundary_id> BoundaryConditionManager::get_neumann_boundary_ids() const {
    std::set<types::boundary_id> ids;
    for (const auto& bc : conditions_) {
        if (is_neumann_bc(bc)) {
            const auto& target = get_target(bc);
            if (target.type == BoundaryTarget::Type::BOUNDARY_ID)
                ids.insert(target.boundary_id);
        }
    }
    return ids;
}

template <int dim>
void BoundaryConditionManager::apply_to_constraints(
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    for (const auto& bc : conditions_) {
        std::visit([&](const auto& specific_bc) {
            using T = std::decay_t<decltype(specific_bc)>;
            
            if constexpr (std::is_same_v<T, FixedBC>)
                apply_fixed_bc(specific_bc, constraints, dof_handler, mapping);
            else if constexpr (std::is_same_v<T, DisplacementBC>)
                apply_displacement_bc(specific_bc, constraints, dof_handler, mapping);
            else if constexpr (std::is_same_v<T, SymmetryBC>)
                apply_symmetry_bc(specific_bc, constraints, dof_handler, mapping);
            else if constexpr (std::is_same_v<T, CyclicSymmetryBC>)
                apply_cyclic_symmetry_bc(specific_bc, constraints, dof_handler, mapping);
            // Elastic support and contact don't add constraints directly
        }, bc);
    }
}

template <int dim>
void BoundaryConditionManager::apply_fixed_bc(
    const FixedBC& bc,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    if (bc.target.type == BoundaryTarget::Type::BOUNDARY_ID) {
        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(
            mapping, dof_handler, bc.target.boundary_id,
            Functions::ZeroFunction<dim>(dim), boundary_values);
        
        for (const auto& [dof, value] : boundary_values) {
            if (!constraints.is_constrained(dof)) {
                constraints.add_line(dof);
                constraints.set_inhomogeneity(dof, value);
            }
        }
    } else if (bc.target.type == BoundaryTarget::Type::POINT ||
               bc.target.type == BoundaryTarget::Type::BOX ||
               bc.target.type == BoundaryTarget::Type::SPHERE ||
               bc.target.type == BoundaryTarget::Type::PLANE) {
        // Find DOFs at matching vertices
        auto target_dofs = find_target_dofs(bc.target, dof_handler, mapping);
        for (const auto& dof : target_dofs) {
            if (!constraints.is_constrained(dof)) {
                constraints.add_line(dof);
                constraints.set_inhomogeneity(dof, 0.0);
            }
        }
    }
}

template <int dim>
void BoundaryConditionManager::apply_displacement_bc(
    const DisplacementBC& bc,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    if (bc.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
        bc.coord_system.type == CoordinateSystem::Type::GLOBAL) {
        // Standard case: global coordinates, boundary ID target
        for (unsigned int d = 0; d < dim; ++d) {
            if (!bc.values[d].has_value()) continue;
            
            std::map<types::global_dof_index, double> boundary_values;
            std::vector<bool> component_mask(dim, false);
            component_mask[d] = true;
            
            Functions::ConstantFunction<dim> const_func(bc.values[d].value(), dim);
            
            VectorTools::interpolate_boundary_values(
                mapping, dof_handler, bc.target.boundary_id,
                const_func, boundary_values, ComponentMask(component_mask));
            
            for (const auto& [dof, value] : boundary_values) {
                if (!constraints.is_constrained(dof)) {
                    constraints.add_line(dof);
                    constraints.set_inhomogeneity(dof, value);
                }
            }
        }
    } else if (bc.coord_system.type != CoordinateSystem::Type::GLOBAL) {
        // Non-global coordinate system requires multi-point constraints
        // For cylindrical/spherical BCs, we need to constrain combinations of DOFs
        
        // Find all vertices matching target
        auto points = find_target_points(bc.target, dof_handler, mapping);
        
        for (const auto& point : points) {
            // Get transformation at this point
            Tensor<2, dim> T = bc.coord_system.get_transformation(point);
            
            // Find DOFs at this point
            std::array<types::global_dof_index, dim> vertex_dofs;
            // ... (implementation would find DOFs near this point)
            
            // Apply constraint in local coordinates
            for (unsigned int local_d = 0; local_d < dim; ++local_d) {
                if (!bc.values[local_d].has_value()) continue;
                
                // u_local[d] = T^T_{dj} * u_global[j] = value
                // This becomes: sum_j T_{jd} * u_j = value
                // Constrain the component with largest coefficient
                
                unsigned int dep_comp = 0;
                double max_coeff = std::abs(T[0][local_d]);
                for (unsigned int j = 1; j < dim; ++j) {
                    if (std::abs(T[j][local_d]) > max_coeff) {
                        max_coeff = std::abs(T[j][local_d]);
                        dep_comp = j;
                    }
                }
                
                // (Implementation continues with MPC constraint)
            }
        }
    }
}

template <int dim>
void BoundaryConditionManager::apply_symmetry_bc(
    const SymmetryBC& bc,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    Tensor<1, dim> n;
    for (unsigned int d = 0; d < dim; ++d)
        n[d] = bc.plane_normal[d];
    
    const auto& fe = dof_handler.get_fe();
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            const auto face = cell->face(f);
            if (!face->at_boundary()) continue;
            
            if (bc.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
                face->boundary_id() != bc.target.boundary_id) continue;
            
            std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());
            cell->face(f)->get_dof_indices(face_dofs);
            
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                std::array<types::global_dof_index, dim> vertex_dofs;
                for (unsigned int d = 0; d < dim; ++d)
                    vertex_dofs[d] = face_dofs[v * dim + d];
                
                // Find component with largest normal coefficient
                unsigned int dep_comp = 0;
                double max_coeff = std::abs(n[0]);
                for (unsigned int d = 1; d < dim; ++d) {
                    if (std::abs(n[d]) > max_coeff) {
                        max_coeff = std::abs(n[d]);
                        dep_comp = d;
                    }
                }
                
                types::global_dof_index dep_dof = vertex_dofs[dep_comp];
                if (constraints.is_constrained(dep_dof)) continue;
                
                // Symmetry constraint: n · u = 0
                // n_0*u_0 + n_1*u_1 + n_2*u_2 = 0
                // Solve for u_dep: u_dep = -sum_{d != dep} (n_d / n_dep) * u_d
                constraints.add_line(dep_dof);
                for (unsigned int d = 0; d < dim; ++d) {
                    if (d == dep_comp) continue;
                    if (std::abs(n[d]) > 1e-12) {
                        constraints.add_entry(dep_dof, vertex_dofs[d], -n[d] / n[dep_comp]);
                    }
                }
                constraints.set_inhomogeneity(dep_dof, 0.0);
            }
        }
    }
}

template <int dim>
void BoundaryConditionManager::apply_cyclic_symmetry_bc(
    const CyclicSymmetryBC& bc,
    AffineConstraints<double>& constraints,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    // Build rotation matrix for sector angle
    Tensor<2, dim> R = rotation_matrix_from_axis_angle(bc.axis_direction, bc.sector_angle);
    
    // Find matching DOF pairs between master and slave boundaries
    // This requires geometric matching of vertices
    
    // (Full implementation would iterate through slave boundary DOFs,
    // find corresponding master DOFs, and create coupling constraints)
}

template <int dim>
std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>>
BoundaryConditionManager::get_elastic_support_matrix_entries(
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim-1>& face_quadrature) const {
    
    std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>> entries;
    const auto& fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    FEFaceValues<dim> fe_face(mapping, fe, face_quadrature,
        update_values | update_JxW_values | update_normal_vectors | update_quadrature_points);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    for (const auto& bc : conditions_) {
        if (!std::holds_alternative<ElasticSupportBC>(bc)) continue;
        const auto& es_bc = std::get<ElasticSupportBC>(bc);
        
        for (const auto& cell : dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;
            
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
                if (!cell->face(f)->at_boundary()) continue;
                
                if (es_bc.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
                    cell->face(f)->boundary_id() != es_bc.target.boundary_id)
                    continue;
                
                fe_face.reinit(cell, f);
                cell->get_dof_indices(local_dof_indices);
                
                for (unsigned int q = 0; q < face_quadrature.size(); ++q) {
                    const double JxW = fe_face.JxW(q);
                    const Tensor<1, dim> normal = fe_face.normal_vector(q);
                    
                    // Compute stiffness in global directions
                    Tensor<1, dim> k;
                    if (es_bc.use_local_directions) {
                        // Transform normal/tangential stiffness to global
                        for (unsigned int d = 0; d < dim; ++d) {
                            k[d] = es_bc.normal_stiffness * normal[d] * normal[d];
                            for (unsigned int e = 0; e < dim; ++e) {
                                if (e != d) {
                                    double tangent_contrib = (d == e) ? 0 : 1;
                                    k[d] += es_bc.tangential_stiffness * tangent_contrib / (dim - 1);
                                }
                            }
                        }
                    } else {
                        for (unsigned int d = 0; d < dim; ++d)
                            k[d] = es_bc.stiffness_per_area[d];
                    }
                    
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        const unsigned int comp_i = fe.system_to_component_index(i).first;
                        if (comp_i >= dim) continue;
                        
                        const double phi_i = fe_face.shape_value(i, q);
                        
                        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                            const unsigned int comp_j = fe.system_to_component_index(j).first;
                            if (comp_i != comp_j) continue;
                            
                            const double phi_j = fe_face.shape_value(j, q);
                            double value = k[comp_i] * phi_i * phi_j * JxW;
                            
                            if (std::abs(value) > 1e-14) {
                                entries.emplace_back(local_dof_indices[i],
                                                    local_dof_indices[j], value);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return entries;
}

template <int dim>
void BoundaryConditionManager::assemble_elastic_support_matrix(
    SparseMatrix<double>& matrix,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const Quadrature<dim-1>& face_quadrature) const {
    
    auto entries = get_elastic_support_matrix_entries(dof_handler, mapping, face_quadrature);
    
    for (const auto& [row, col, value] : entries) {
        matrix.add(row, col, value);
    }
}

template <int dim>
void BoundaryConditionManager::get_contact_contributions(
    const Vector<double>& current_solution,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    Vector<double>& contact_forces,
    std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>>& contact_stiffness) const {
    
    const auto& fe = dof_handler.get_fe();
    
    // Clear outputs
    contact_forces = 0;
    contact_stiffness.clear();
    active_contact_dofs_.clear();
    
    for (const auto& bc : conditions_) {
        if (!std::holds_alternative<FrictionlessSupportBC>(bc)) continue;
        const auto& contact_bc = std::get<FrictionlessSupportBC>(bc);
        
        for (const auto& cell : dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;
            
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
                if (!cell->face(f)->at_boundary()) continue;
                
                if (contact_bc.target.type == BoundaryTarget::Type::BOUNDARY_ID &&
                    cell->face(f)->boundary_id() != contact_bc.target.boundary_id)
                    continue;
                
                // Get face vertices and their DOFs
                for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                    const auto vertex = cell->face(f)->vertex(v);
                    
                    // Get displacement at this vertex
                    std::vector<types::global_dof_index> local_dofs(fe.n_dofs_per_cell());
                    cell->get_dof_indices(local_dofs);
                    
                    Tensor<1, dim> displacement;
                    for (unsigned int d = 0; d < dim; ++d) {
                        // Find DOF for this vertex and component
                        // (Simplified - actual implementation needs vertex DOF mapping)
                        displacement[d] = 0; // current_solution[vertex_dof];
                    }
                    
                    // Check contact
                    double normal_disp = displacement * contact_bc.surface_normal;
                    double gap = contact_bc.initial_gap - normal_disp;
                    
                    if (gap < 0) {
                        // Contact active - add penalty force and stiffness
                        double penalty_force = -contact_bc.penalty_stiffness * gap;
                        
                        // (Add contributions to contact_forces and contact_stiffness)
                    }
                }
            }
        }
    }
}

template <int dim>
std::vector<types::global_dof_index> BoundaryConditionManager::find_target_dofs(
    const BoundaryTarget& target,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    std::optional<unsigned int> component) const {
    
    std::vector<types::global_dof_index> dofs;
    const auto& fe = dof_handler.get_fe();
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            Point<dim> vertex = cell->vertex(v);
            Point<3> vertex_3d;
            for (unsigned int d = 0; d < dim && d < 3; ++d)
                vertex_3d[d] = vertex[d];
            
            if (target.contains_point(vertex_3d)) {
                std::vector<types::global_dof_index> local_dofs(fe.n_dofs_per_cell());
                cell->get_dof_indices(local_dofs);
                
                for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) {
                    auto [comp, index] = fe.system_to_component_index(i);
                    if (component.has_value() && comp != component.value())
                        continue;
                    
                    // Check if this DOF is at the vertex
                    // (This depends on FE type - for Q1, DOFs are at vertices)
                    if (fe.has_support_on_face(i, 0)) {  // Simplified check
                        dofs.push_back(local_dofs[i]);
                    }
                }
            }
        }
    }
    
    // Remove duplicates
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    
    return dofs;
}

template <int dim>
std::vector<Point<dim>> BoundaryConditionManager::find_target_points(
    const BoundaryTarget& target,
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    std::vector<Point<dim>> points;
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            Point<dim> vertex = cell->vertex(v);
            Point<3> vertex_3d;
            for (unsigned int d = 0; d < dim && d < 3; ++d)
                vertex_3d[d] = vertex[d];
            
            if (target.contains_point(vertex_3d)) {
                points.push_back(vertex);
            }
        }
    }
    
    // Remove duplicates (with tolerance)
    auto nearly_equal = [](const Point<dim>& a, const Point<dim>& b) {
        return a.distance(b) < 1e-10;
    };
    
    std::vector<Point<dim>> unique_points;
    for (const auto& p : points) {
        bool found = false;
        for (const auto& up : unique_points) {
            if (nearly_equal(p, up)) {
                found = true;
                break;
            }
        }
        if (!found) unique_points.push_back(p);
    }
    
    return unique_points;
}

template <int dim>
std::vector<std::string> BoundaryConditionManager::validate(
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping) const {
    
    std::vector<std::string> errors;
    
    for (size_t i = 0; i < conditions_.size(); ++i) {
        const auto& bc = conditions_[i];
        const auto& target = get_target(bc);
        
        if (target.type == BoundaryTarget::Type::BOUNDARY_ID) {
            // Check if boundary ID exists in mesh
            bool found = false;
            for (const auto& cell : dof_handler.active_cell_iterators()) {
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
                    if (cell->face(f)->at_boundary() &&
                        cell->face(f)->boundary_id() == target.boundary_id) {
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            
            if (!found) {
                errors.push_back("BC[" + std::to_string(i) + "]: boundary_id " +
                               std::to_string(target.boundary_id) + " not found in mesh");
            }
        }
    }
    
    return errors;
}

// ============================================================================
// Helper Functions
// ============================================================================

void add_mpc_to_constraints(
    AffineConstraints<double>& constraints,
    const std::vector<std::pair<types::global_dof_index, double>>& terms,
    double rhs) {
    
    if (terms.empty()) return;
    
    // Find term with largest absolute coefficient
    size_t dep_idx = 0;
    double max_coeff = std::abs(terms[0].second);
    for (size_t i = 1; i < terms.size(); ++i) {
        if (std::abs(terms[i].second) > max_coeff) {
            max_coeff = std::abs(terms[i].second);
            dep_idx = i;
        }
    }
    
    types::global_dof_index dep_dof = terms[dep_idx].first;
    double dep_coeff = terms[dep_idx].second;
    
    if (constraints.is_constrained(dep_dof)) return;
    
    constraints.add_line(dep_dof);
    
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i == dep_idx) continue;
        if (std::abs(terms[i].second) > 1e-14) {
            constraints.add_entry(dep_dof, terms[i].first, -terms[i].second / dep_coeff);
        }
    }
    
    constraints.set_inhomogeneity(dep_dof, rhs / dep_coeff);
}

Tensor<2, 3> rotation_matrix_from_axis_angle(const Tensor<1, 3>& axis, double angle) {
    Tensor<1, 3> k = axis / axis.norm();
    double c = std::cos(angle);
    double s = std::sin(angle);
    
    Tensor<2, 3> R;
    
    // Rodrigues' rotation formula
    R[0][0] = c + k[0]*k[0]*(1-c);
    R[0][1] = k[0]*k[1]*(1-c) - k[2]*s;
    R[0][2] = k[0]*k[2]*(1-c) + k[1]*s;
    
    R[1][0] = k[1]*k[0]*(1-c) + k[2]*s;
    R[1][1] = c + k[1]*k[1]*(1-c);
    R[1][2] = k[1]*k[2]*(1-c) - k[0]*s;
    
    R[2][0] = k[2]*k[0]*(1-c) - k[1]*s;
    R[2][1] = k[2]*k[1]*(1-c) + k[0]*s;
    R[2][2] = c + k[2]*k[2]*(1-c);
    
    return R;
}

Tensor<2, 3> rotation_matrix_aligning_z_with(const Tensor<1, 3>& direction) {
    Tensor<1, 3> z = direction / direction.norm();
    
    // Find a vector not parallel to z
    Tensor<1, 3> temp;
    if (std::abs(z[0]) < 0.9) {
        temp[0] = 1; temp[1] = 0; temp[2] = 0;
    } else {
        temp[0] = 0; temp[1] = 1; temp[2] = 0;
    }
    
    // x = normalize(temp - (temp·z)z)
    Tensor<1, 3> x = temp - (temp * z) * z;
    x /= x.norm();
    
    // y = z × x
    Tensor<1, 3> y;
    y[0] = z[1]*x[2] - z[2]*x[1];
    y[1] = z[2]*x[0] - z[0]*x[2];
    y[2] = z[0]*x[1] - z[1]*x[0];
    
    Tensor<2, 3> R;
    for (unsigned int d = 0; d < 3; ++d) {
        R[d][0] = x[d];
        R[d][1] = y[d];
        R[d][2] = z[d];
    }
    
    return R;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void BoundaryConditionManager::apply_to_constraints<3>(
    AffineConstraints<double>&, const DoFHandler<3>&, const Mapping<3>&) const;

template void BoundaryConditionManager::apply_to_constraints<2>(
    AffineConstraints<double>&, const DoFHandler<2>&, const Mapping<2>&) const;

template std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>>
BoundaryConditionManager::get_elastic_support_matrix_entries<3>(
    const DoFHandler<3>&, const Mapping<3>&, const Quadrature<2>&) const;

template std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>>
BoundaryConditionManager::get_elastic_support_matrix_entries<2>(
    const DoFHandler<2>&, const Mapping<2>&, const Quadrature<1>&) const;

template void BoundaryConditionManager::assemble_elastic_support_matrix<3>(
    SparseMatrix<double>&, const DoFHandler<3>&, const Mapping<3>&, const Quadrature<2>&) const;

template void BoundaryConditionManager::get_contact_contributions<3>(
    const Vector<double>&, const DoFHandler<3>&, const Mapping<3>&,
    Vector<double>&, std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>>&) const;

template std::vector<types::global_dof_index> BoundaryConditionManager::find_target_dofs<3>(
    const BoundaryTarget&, const DoFHandler<3>&, const Mapping<3>&, std::optional<unsigned int>) const;

template std::vector<Point<3>> BoundaryConditionManager::find_target_points<3>(
    const BoundaryTarget&, const DoFHandler<3>&, const Mapping<3>&) const;

template std::vector<std::string> BoundaryConditionManager::validate<3>(
    const DoFHandler<3>&, const Mapping<3>&) const;

} // namespace FEA
