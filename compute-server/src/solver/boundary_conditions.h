/**
 * @file boundary_conditions.h
 * @brief Comprehensive boundary condition system for FEA
 * 
 * Supports:
 * - Fixed (fully constrained)
 * - Prescribed displacement (with component selection)
 * - Symmetry planes
 * - Elastic (spring) supports
 * - Frictionless contact with rigid surfaces
 */

#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

#include <variant>
#include <vector>
#include <array>
#include <optional>
#include <string>
#include <set>
#include <functional>
#include <memory>
#include <map>

namespace FEA {

using namespace dealii;

// ============================================================================
// Boundary Target Specification
// ============================================================================

/**
 * @brief Defines how to identify which part of the geometry a BC applies to
 * 
 * Multiple selection methods are supported to accommodate different
 * mesh formats and modeling workflows.
 */
struct BoundaryTarget {
    enum class Type {
        BOUNDARY_ID,      // Face boundary indicator (set in mesh or by deal.ii)
        MATERIAL_ID,      // Cell material ID (region)
        POINT,            // Single point (finds nearest node)
        NODE_SET,         // Named node set from mesh file
        FACE_SET,         // Named face set from mesh file
        BOX,              // All nodes within axis-aligned bounding box
        SPHERE,           // All nodes within a sphere
        CYLINDER,         // All nodes within a cylinder
        PLANE             // All nodes on/near a plane
    };
    
    Type type = Type::BOUNDARY_ID;
    
    // For BOUNDARY_ID
    unsigned int boundary_id = 0;
    
    // For MATERIAL_ID
    unsigned int material_id = 0;
    
    // For POINT
    Point<3> point;
    double point_tolerance = 1e-10;
    
    // For NODE_SET / FACE_SET
    std::string set_name;
    
    // For BOX
    Point<3> box_min, box_max;
    
    // For SPHERE
    Point<3> sphere_center;
    double sphere_radius = 0;
    
    // For CYLINDER
    Point<3> cylinder_point;
    Tensor<1, 3> cylinder_axis;
    double cylinder_radius = 0;
    double cylinder_length = std::numeric_limits<double>::infinity();
    
    // For PLANE
    Point<3> plane_point;
    Tensor<1, 3> plane_normal;
    double plane_tolerance = 1e-6;
    
    // Factory methods for convenient construction
    static BoundaryTarget from_boundary_id(unsigned int id);
    static BoundaryTarget from_material_id(unsigned int id);
    static BoundaryTarget from_point(const Point<3>& p, double tol = 1e-10);
    static BoundaryTarget from_node_set(const std::string& name);
    static BoundaryTarget from_face_set(const std::string& name);
    static BoundaryTarget from_box(const Point<3>& min, const Point<3>& max);
    static BoundaryTarget from_sphere(const Point<3>& center, double radius);
    static BoundaryTarget from_cylinder(const Point<3>& base, const Tensor<1, 3>& axis,
                                        double radius, double length = std::numeric_limits<double>::infinity());
    static BoundaryTarget from_plane(const Point<3>& point, 
                                     const Tensor<1, 3>& normal, 
                                     double tol = 1e-6);
    
    // Check if a point matches this target
    bool contains_point(const Point<3>& p) const;
    
    std::string to_string() const;
};

// ============================================================================
// Coordinate System Specification
// ============================================================================

/**
 * @brief Coordinate system for specifying BC values
 * 
 * Allows BCs to be specified in cylindrical, spherical, or local
 * coordinate systems, which is essential for axisymmetric problems
 * and components with complex geometry.
 */
struct CoordinateSystem {
    enum class Type { GLOBAL, CYLINDRICAL, SPHERICAL, LOCAL };
    
    Type type = Type::GLOBAL;
    
    // For CYLINDRICAL: r, θ, z coordinates about an axis
    Point<3> axis_origin;
    Tensor<1, 3> axis_direction;
    
    // For SPHERICAL: r, θ, φ coordinates about a center
    Point<3> center;
    
    // For LOCAL: arbitrary rotated coordinate system
    Tensor<2, 3> rotation_matrix;
    Point<3> origin;
    
    /**
     * @brief Get transformation matrix from local to global at a point
     * 
     * For cylindrical/spherical systems, the transformation depends
     * on the location of the point.
     */
    Tensor<2, 3> get_transformation(const Point<3>& at_point) const;
    
    /**
     * @brief Transform displacement from local to global coordinates
     */
    Tensor<1, 3> to_global(const Tensor<1, 3>& local_disp, const Point<3>& at_point) const;
    
    /**
     * @brief Transform displacement from global to local coordinates
     */
    Tensor<1, 3> to_local(const Tensor<1, 3>& global_disp, const Point<3>& at_point) const;
    
    // Factory methods
    static CoordinateSystem global();
    static CoordinateSystem cylindrical(const Point<3>& origin, const Tensor<1, 3>& axis);
    static CoordinateSystem spherical(const Point<3>& center);
    static CoordinateSystem local(const Tensor<2, 3>& R, const Point<3>& origin = Point<3>());
};

// ============================================================================
// Boundary Condition Types
// ============================================================================

/**
 * @brief Fixed boundary condition - all DOFs set to zero
 * 
 * Equivalent to a fully clamped support where no motion is allowed.
 */
struct FixedBC {
    BoundaryTarget target;
    std::string description;
    
    FixedBC() = default;
    explicit FixedBC(const BoundaryTarget& t, const std::string& desc = "")
        : target(t), description(desc) {}
};

/**
 * @brief Prescribed displacement - can constrain individual components
 * 
 * std::nullopt for a component means that direction is unconstrained.
 * Values can be specified in a local coordinate system.
 */
struct DisplacementBC {
    BoundaryTarget target;
    
    // std::nullopt means unconstrained in that direction
    std::array<std::optional<double>, 3> values = {0.0, 0.0, 0.0};
    
    CoordinateSystem coord_system;
    std::string description;
    
    DisplacementBC() = default;
    
    // Factory methods for common cases
    static DisplacementBC fixed(const BoundaryTarget& target);
    static DisplacementBC x_only(const BoundaryTarget& target, double ux);
    static DisplacementBC y_only(const BoundaryTarget& target, double uy);
    static DisplacementBC z_only(const BoundaryTarget& target, double uz);
    static DisplacementBC xy_only(const BoundaryTarget& target, double ux, double uy);
    static DisplacementBC xz_only(const BoundaryTarget& target, double ux, double uz);
    static DisplacementBC yz_only(const BoundaryTarget& target, double uy, double uz);
    static DisplacementBC full(const BoundaryTarget& target, double ux, double uy, double uz);
    
    // With coordinate system
    static DisplacementBC radial_only(const BoundaryTarget& target, double ur,
                                      const Point<3>& axis_origin, const Tensor<1, 3>& axis);
    
    bool is_x_constrained() const { return values[0].has_value(); }
    bool is_y_constrained() const { return values[1].has_value(); }
    bool is_z_constrained() const { return values[2].has_value(); }
    int num_constrained() const;
    bool is_fully_constrained() const { return num_constrained() == 3; }
};

/**
 * @brief Symmetry plane - constrains normal displacement to zero
 * 
 * Used when the model represents half, quarter, or eighth of a
 * symmetric structure. Only the normal component is constrained.
 */
struct SymmetryBC {
    BoundaryTarget target;
    Tensor<1, 3> plane_normal;
    std::string description;
    
    SymmetryBC() = default;
    SymmetryBC(const BoundaryTarget& t, const Tensor<1, 3>& normal, 
               const std::string& desc = "")
        : target(t), plane_normal(normal / normal.norm()), description(desc) {}
    
    // Factory methods for axis-aligned symmetry planes
    static SymmetryBC x_plane(const BoundaryTarget& target);  // normal = (1,0,0)
    static SymmetryBC y_plane(const BoundaryTarget& target);  // normal = (0,1,0)
    static SymmetryBC z_plane(const BoundaryTarget& target);  // normal = (0,0,1)
};

/**
 * @brief Elastic (spring) support - distributed springs to ground
 * 
 * Models foundation support, soft mounts, or compliance in supports.
 * Stiffness is specified per unit area [N/m³].
 */
struct ElasticSupportBC {
    BoundaryTarget target;
    
    // Global stiffness components [N/m³]
    Tensor<1, 3> stiffness_per_area;
    
    // Alternative: local normal/tangential stiffnesses
    bool use_local_directions = false;
    double normal_stiffness = 0;      // [N/m³] - perpendicular to surface
    double tangential_stiffness = 0;  // [N/m³] - in-plane
    
    std::string description;
    
    ElasticSupportBC() = default;
    ElasticSupportBC(const BoundaryTarget& t, const Tensor<1, 3>& k, 
                     const std::string& desc = "")
        : target(t), stiffness_per_area(k), description(desc) {}
    
    // Factory methods
    static ElasticSupportBC uniform(const BoundaryTarget& target, double k);
    static ElasticSupportBC normal_only(const BoundaryTarget& target, double k_n);
    static ElasticSupportBC tangential_only(const BoundaryTarget& target, double k_t);
    static ElasticSupportBC anisotropic(const BoundaryTarget& target, 
                                        double kx, double ky, double kz);
};

/**
 * @brief Frictionless support - unilateral contact with rigid surface
 * 
 * Models contact with a rigid surface where:
 * - Normal gap cannot be negative (no penetration)
 * - No friction (tangential motion is free)
 * - Supports can separate (gap can open)
 * 
 * Implemented using penalty method for simplicity.
 */
struct FrictionlessSupportBC {
    BoundaryTarget target;
    Tensor<1, 3> surface_normal;      // Outward normal of rigid surface
    double initial_gap = 0;            // Initial distance from surface [m]
    double penalty_stiffness = 1e12;   // Contact stiffness [N/m³]
    std::string description;
    
    FrictionlessSupportBC() = default;
    FrictionlessSupportBC(const BoundaryTarget& t, const Tensor<1, 3>& normal,
                          double gap = 0, const std::string& desc = "")
        : target(t), surface_normal(normal / normal.norm()),
          initial_gap(gap), description(desc) {}
    
    // Check if contact is active given current displacement
    bool is_contact_active(const Tensor<1, 3>& displacement) const;
    
    // Get contact force given displacement (penalty method)
    Tensor<1, 3> get_contact_force(const Tensor<1, 3>& displacement) const;
};

/**
 * @brief Cyclic symmetry boundary condition
 * 
 * For rotationally periodic structures (turbine blades, gears).
 * Links DOFs on one sector boundary to another.
 */
struct CyclicSymmetryBC {
    BoundaryTarget master_target;
    BoundaryTarget slave_target;
    Point<3> axis_origin;
    Tensor<1, 3> axis_direction;
    double sector_angle;  // [radians]
    std::string description;
    
    CyclicSymmetryBC() = default;
};

// ============================================================================
// Union Type for All Boundary Conditions
// ============================================================================

using BoundaryCondition = std::variant<
    FixedBC,
    DisplacementBC,
    SymmetryBC,
    ElasticSupportBC,
    FrictionlessSupportBC,
    CyclicSymmetryBC
>;

// Accessor functions for variant type
const BoundaryTarget& get_target(const BoundaryCondition& bc);
const std::string& get_description(const BoundaryCondition& bc);
bool is_dirichlet_bc(const BoundaryCondition& bc);
bool is_neumann_bc(const BoundaryCondition& bc);
bool is_nonlinear_bc(const BoundaryCondition& bc);
bool requires_coupling(const BoundaryCondition& bc);

// ============================================================================
// Boundary Condition Manager
// ============================================================================

/**
 * @brief Manages and applies boundary conditions to the FE system
 * 
 * Handles:
 * - Collection and validation of boundary conditions
 * - Application to deal.II constraint matrices
 * - Elastic support matrix contributions
 * - Contact iteration for nonlinear BCs
 */
class BoundaryConditionManager {
public:
    BoundaryConditionManager();
    ~BoundaryConditionManager() = default;
    
    // ===== Condition Management =====
    
    void add_condition(const BoundaryCondition& bc);
    void add_conditions(const std::vector<BoundaryCondition>& bcs);
    void remove_condition(size_t index);
    void clear();
    
    const std::vector<BoundaryCondition>& get_conditions() const { return conditions_; }
    size_t size() const { return conditions_.size(); }
    bool empty() const { return conditions_.empty(); }
    
    // ===== Constraint Application =====
    
    /**
     * @brief Apply all Dirichlet-type BCs to constraint matrix
     */
    template <int dim>
    void apply_to_constraints(
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    /**
     * @brief Get elastic support matrix contributions
     * 
     * Returns triplets (row, col, value) to be added to system matrix.
     */
    template <int dim>
    std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>>
    get_elastic_support_matrix_entries(
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim-1>& face_quadrature) const;
    
    /**
     * @brief Assemble elastic support contributions into sparse matrix
     */
    template <int dim>
    void assemble_elastic_support_matrix(
        SparseMatrix<double>& matrix,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Quadrature<dim-1>& face_quadrature) const;
    
    /**
     * @brief Get contact force contributions (for iterative contact)
     */
    template <int dim>
    void get_contact_contributions(
        const Vector<double>& current_solution,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        Vector<double>& contact_forces,
        std::vector<std::tuple<types::global_dof_index, types::global_dof_index, double>>& contact_stiffness) const;
    
    // ===== Query Methods =====
    
    bool has_contact_conditions() const;
    bool has_elastic_support_conditions() const;
    bool has_cyclic_symmetry() const;
    bool has_nonlinear_conditions() const;
    
    std::set<types::boundary_id> get_dirichlet_boundary_ids() const;
    std::set<types::boundary_id> get_neumann_boundary_ids() const;
    
    // ===== Validation =====
    
    /**
     * @brief Validate boundary conditions against mesh
     */
    template <int dim>
    std::vector<std::string> validate(
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
private:
    std::vector<BoundaryCondition> conditions_;
    
    // Cache for contact iteration
    mutable std::set<types::global_dof_index> active_contact_dofs_;
    
    // ===== Internal Application Methods =====
    
    template <int dim>
    std::vector<types::global_dof_index> find_target_dofs(
        const BoundaryTarget& target,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        std::optional<unsigned int> component = std::nullopt) const;
    
    template <int dim>
    std::vector<Point<dim>> find_target_points(
        const BoundaryTarget& target,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_fixed_bc(
        const FixedBC& bc,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_displacement_bc(
        const DisplacementBC& bc,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_symmetry_bc(
        const SymmetryBC& bc,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_cyclic_symmetry_bc(
        const CyclicSymmetryBC& bc,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Create constraint from multi-point constraint equation
 * 
 * Constrains: c0*u0 + c1*u1 + ... = rhs
 * Solved for the DOF with largest coefficient.
 */
void add_mpc_to_constraints(
    AffineConstraints<double>& constraints,
    const std::vector<std::pair<types::global_dof_index, double>>& terms,
    double rhs = 0);

/**
 * @brief Build rotation matrix from axis angle
 */
Tensor<2, 3> rotation_matrix_from_axis_angle(
    const Tensor<1, 3>& axis,
    double angle);

/**
 * @brief Build rotation matrix aligning z-axis with given direction
 */
Tensor<2, 3> rotation_matrix_aligning_z_with(const Tensor<1, 3>& direction);

} // namespace FEA

#endif // BOUNDARY_CONDITIONS_H
