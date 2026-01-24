#ifndef CONSTRAINT_BASE_H
#define CONSTRAINT_BASE_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>

#include "../boundary_conditions.h"

#include <variant>
#include <vector>
#include <array>
#include <string>
#include <map>
#include <memory>
#include <optional>
#include <numeric>

namespace FEA {

using namespace dealii;

// ============================================================================
// SPRING ELEMENTS
// ============================================================================

/**
 * @brief Spring connection to ground (fixed reference)
 * 
 * Applies a spring stiffness from nodes to a fixed ground point.
 * Can have different stiffness in each direction and optional preload.
 */
struct SpringToGroundConnection {
    BoundaryTarget target;              ///< Nodes to connect to ground
    Tensor<1, 3> stiffness;             ///< [N/m] stiffness in each direction
    Tensor<1, 3> preload_force;         ///< [N] positive = tension
    bool use_local_coords = false;      ///< Use local coordinate system
    Tensor<2, 3> local_to_global;       ///< Transformation matrix
    std::string description;
    
    SpringToGroundConnection() : preload_force() {
        local_to_global = unit_symmetric_tensor<3>();
    }
    
    /**
     * @brief Create uniform spring with same stiffness in all directions
     */
    static SpringToGroundConnection uniform(const BoundaryTarget& target, double k) {
        SpringToGroundConnection conn;
        conn.target = target;
        conn.stiffness = Tensor<1, 3>({k, k, k});
        conn.description = "Uniform spring to ground";
        return conn;
    }
    
    /**
     * @brief Create spring with different stiffness in each direction
     */
    static SpringToGroundConnection directional(const BoundaryTarget& target,
                                                 double kx, double ky, double kz) {
        SpringToGroundConnection conn;
        conn.target = target;
        conn.stiffness = Tensor<1, 3>({kx, ky, kz});
        conn.description = "Directional spring to ground";
        return conn;
    }
    
    /**
     * @brief Create spring with normal stiffness (common for elastic supports)
     */
    static SpringToGroundConnection normal_only(const BoundaryTarget& target,
                                                 double k_normal,
                                                 const Tensor<1, 3>& normal) {
        SpringToGroundConnection conn;
        conn.target = target;
        conn.use_local_coords = true;
        
        // Build local system with normal as Z
        Tensor<1, 3> n = normal / normal.norm();
        Tensor<1, 3> temp = (std::abs(n[0]) < 0.9) ?
            Tensor<1, 3>({1, 0, 0}) : Tensor<1, 3>({0, 1, 0});
        Tensor<1, 3> t1 = temp - (temp * n) * n;
        t1 /= t1.norm();
        Tensor<1, 3> t2 = cross_product_3d(n, t1);
        
        for (unsigned int i = 0; i < 3; ++i) {
            conn.local_to_global[i][0] = t1[i];
            conn.local_to_global[i][1] = t2[i];
            conn.local_to_global[i][2] = n[i];
        }
        
        conn.stiffness = Tensor<1, 3>({0, 0, k_normal});
        conn.description = "Normal spring to ground";
        return conn;
    }
};

/**
 * @brief Spring connection between two points
 * 
 * Creates an axial spring element between two mesh nodes.
 * Can include lateral stiffness and preload.
 */
struct SpringConnection {
    Point<3> point_a;                   ///< First connection point
    Point<3> point_b;                   ///< Second connection point
    double axial_stiffness;             ///< [N/m] stiffness along spring axis
    double preload_force = 0.0;         ///< [N] preload (positive = tension)
    double lateral_stiffness = 0.0;     ///< [N/m] stiffness perpendicular to axis
    double node_search_tolerance = 1e-6;///< Tolerance for finding mesh nodes
    std::string description;
    
    SpringConnection() = default;
    SpringConnection(const Point<3>& a, const Point<3>& b, double k)
        : point_a(a), point_b(b), axial_stiffness(k) {}
    
    /**
     * @brief Get unit vector along spring axis
     */
    Tensor<1, 3> get_axis() const {
        Tensor<1, 3> axis;
        for (unsigned int d = 0; d < 3; ++d)
            axis[d] = point_b[d] - point_a[d];
        double len = axis.norm();
        return (len > 1e-14) ? axis / len : Tensor<1, 3>();
    }
    
    /**
     * @brief Get spring length (undeformed)
     */
    double get_length() const {
        return point_a.distance(point_b);
    }
    
    /**
     * @brief Create spring with specified stiffness and optional preload
     */
    static SpringConnection create(const Point<3>& a, const Point<3>& b,
                                   double k, double preload = 0.0) {
        SpringConnection conn;
        conn.point_a = a;
        conn.point_b = b;
        conn.axial_stiffness = k;
        conn.preload_force = preload;
        conn.description = "Two-point spring";
        return conn;
    }
};

/**
 * @brief 6-DOF bushing connection between two points
 * 
 * Models a rubber mount or flexible coupling with translational
 * and rotational stiffness components. Can specify full 6x6 stiffness
 * matrix for coupled behavior.
 */
struct BushingConnection {
    Point<3> point_a;                   ///< First connection point
    Point<3> point_b;                   ///< Second connection point
    Tensor<1, 3> translational_stiffness;   ///< [N/m] in each direction
    Tensor<1, 3> rotational_stiffness;      ///< [N·m/rad] about each axis
    std::optional<FullMatrix<double>> full_stiffness_matrix;  ///< Optional 6x6 coupling
    Tensor<1, 3> translational_preload;     ///< [N] preload forces
    Tensor<1, 3> rotational_preload;        ///< [N·m] preload moments
    Tensor<2, 3> orientation;               ///< Local to global rotation
    double node_search_tolerance = 1e-6;
    std::string description;
    
    BushingConnection() {
        orientation = unit_symmetric_tensor<3>();
    }
    
    /**
     * @brief Set bushing orientation from primary axis direction
     */
    void set_orientation_from_axis(const Tensor<1, 3>& x_dir) {
        Tensor<1, 3> x = x_dir / x_dir.norm();
        Tensor<1, 3> temp = (std::abs(x[0]) < 0.9) ?
            Tensor<1, 3>({1, 0, 0}) : Tensor<1, 3>({0, 1, 0});
        Tensor<1, 3> y = temp - (temp * x) * x;
        y /= y.norm();
        Tensor<1, 3> z = cross_product_3d(x, y);
        
        for (unsigned int i = 0; i < 3; ++i) {
            orientation[i][0] = x[i];
            orientation[i][1] = y[i];
            orientation[i][2] = z[i];
        }
    }
    
    /**
     * @brief Create isotropic bushing with same stiffness in all directions
     */
    static BushingConnection isotropic(const Point<3>& a, const Point<3>& b,
                                        double k_trans, double k_rot) {
        BushingConnection conn;
        conn.point_a = a;
        conn.point_b = b;
        conn.translational_stiffness = Tensor<1, 3>({k_trans, k_trans, k_trans});
        conn.rotational_stiffness = Tensor<1, 3>({k_rot, k_rot, k_rot});
        conn.description = "Isotropic bushing";
        return conn;
    }
    
    /**
     * @brief Create cylindrical bushing (stiff axially, soft radially)
     */
    static BushingConnection cylindrical(const Point<3>& a, const Point<3>& b,
                                          double k_axial, double k_radial,
                                          double k_torsion, double k_bending) {
        BushingConnection conn;
        conn.point_a = a;
        conn.point_b = b;
        
        // Compute axis from point_a to point_b
        Tensor<1, 3> axis;
        for (unsigned int d = 0; d < 3; ++d)
            axis[d] = b[d] - a[d];
        if (axis.norm() > 1e-14) {
            conn.set_orientation_from_axis(axis);
        }
        
        // X = axial, Y,Z = radial
        conn.translational_stiffness = Tensor<1, 3>({k_axial, k_radial, k_radial});
        conn.rotational_stiffness = Tensor<1, 3>({k_torsion, k_bending, k_bending});
        conn.description = "Cylindrical bushing";
        return conn;
    }
};

// ============================================================================
// RIGID CONNECTIONS
// ============================================================================

/**
 * @brief Rigid or distributing connection (RBE2/RBE3)
 * 
 * RBE2 (is_rigid=true): Slaves follow master with rigid body motion
 * RBE3 (is_rigid=false): Master motion is weighted average of slaves
 */
struct RigidConnection {
    Point<3> master_point;              ///< Reference/control point
    BoundaryTarget slave_target;        ///< Connected nodes
    bool is_rigid = true;               ///< true=RBE2, false=RBE3
    std::vector<double> slave_weights;  ///< Weights for RBE3 (optional)
    std::array<bool, 3> coupled_dofs = {true, true, true};  ///< Which DOFs to couple
    double node_search_tolerance = 1e-6;
    std::string description;
    
    RigidConnection() = default;
    
    /**
     * @brief Create RBE2 rigid link (slaves constrained to master)
     * 
     * Slaves follow the rigid body motion of the master point.
     * Used for: rigid spiders, kinematic coupling, load application points
     */
    static RigidConnection rbe2(const Point<3>& master, const BoundaryTarget& slaves) {
        RigidConnection conn;
        conn.master_point = master;
        conn.slave_target = slaves;
        conn.is_rigid = true;
        conn.description = "RBE2 rigid link";
        return conn;
    }
    
    /**
     * @brief Create RBE3 distributing link (master follows weighted slaves)
     * 
     * Master motion is computed as weighted average of slave motions.
     * Used for: load distribution, averaging connections
     */
    static RigidConnection rbe3(const Point<3>& master, const BoundaryTarget& slaves) {
        RigidConnection conn;
        conn.master_point = master;
        conn.slave_target = slaves;
        conn.is_rigid = false;
        conn.description = "RBE3 distributing link";
        return conn;
    }
    
    /**
     * @brief Create RBE3 with specified weights
     */
    static RigidConnection rbe3_weighted(const Point<3>& master,
                                          const BoundaryTarget& slaves,
                                          const std::vector<double>& weights) {
        RigidConnection conn;
        conn.master_point = master;
        conn.slave_target = slaves;
        conn.is_rigid = false;
        conn.slave_weights = weights;
        conn.description = "RBE3 weighted distributing link";
        return conn;
    }
};

/**
 * @brief Tied contact connection (bonded surfaces)
 * 
 * Couples displacement DOFs between master and slave surfaces.
 * Slave nodes within tolerance are constrained to follow the closest master node.
 */
struct TiedConnection {
    BoundaryTarget master_surface;      ///< Master surface (retained DOFs)
    BoundaryTarget slave_surface;       ///< Slave surface (constrained DOFs)
    double tie_tolerance;               ///< Maximum gap to tie
    double search_radius = 0.0;         ///< Search radius (0 = automatic)
    bool position_tolerance_check = true;  ///< Check initial position mismatch
    std::string description;
    
    TiedConnection() = default;
    TiedConnection(const BoundaryTarget& master, const BoundaryTarget& slave,
                   double tolerance)
        : master_surface(master), slave_surface(slave), tie_tolerance(tolerance) {
        description = "Tied contact";
    }
    
    /**
     * @brief Create tied connection for bonded surfaces
     */
    static TiedConnection bonded(const BoundaryTarget& master,
                                  const BoundaryTarget& slave,
                                  double tolerance) {
        TiedConnection conn;
        conn.master_surface = master;
        conn.slave_surface = slave;
        conn.tie_tolerance = tolerance;
        conn.description = "Bonded interface";
        return conn;
    }
};

/**
 * @brief Directional coupling (DOFs coupled in specified direction)
 * 
 * Constrains nodes to move together in a specified direction only.
 * Useful for modeling planar motion constraints or guide rails.
 */
struct DirectionalCoupling {
    BoundaryTarget target;              ///< Nodes to couple
    Tensor<1, 3> direction;             ///< Coupling direction (normalized)
    Point<3> reference_point;           ///< Reference for computing constraint
    double reference_tolerance = 1e-6;
    std::string description;
    
    /**
     * @brief Create coupling in X direction
     */
    static DirectionalCoupling x_direction(const BoundaryTarget& target) {
        DirectionalCoupling conn;
        conn.target = target;
        conn.direction = Tensor<1, 3>({1, 0, 0});
        conn.description = "X-directional coupling";
        return conn;
    }
    
    /**
     * @brief Create coupling in arbitrary direction
     */
    static DirectionalCoupling in_direction(const BoundaryTarget& target,
                                             const Tensor<1, 3>& dir) {
        DirectionalCoupling conn;
        conn.target = target;
        conn.direction = dir / dir.norm();
        conn.description = "Directional coupling";
        return conn;
    }
};

/**
 * @brief Cylindrical coupling (radial, circumferential, axial)
 * 
 * Couples nodes in cylindrical coordinates about a specified axis.
 * Useful for axisymmetric constraints and rotating machinery.
 */
struct CylindricalCoupling {
    BoundaryTarget target;              ///< Nodes to couple
    Point<3> axis_point;                ///< Point on the axis
    Tensor<1, 3> axis_direction;        ///< Axis direction (normalized)
    bool couple_radial = false;         ///< Couple radial displacement
    bool couple_circumferential = false;///< Couple circumferential displacement
    bool couple_axial = false;          ///< Couple axial displacement
    std::string description;
    
    /**
     * @brief Create coupling for radial motion only
     */
    static CylindricalCoupling radial_only(const BoundaryTarget& target,
                                            const Point<3>& axis_pt,
                                            const Tensor<1, 3>& axis_dir) {
        CylindricalCoupling conn;
        conn.target = target;
        conn.axis_point = axis_pt;
        conn.axis_direction = axis_dir / axis_dir.norm();
        conn.couple_radial = true;
        conn.description = "Radial cylindrical coupling";
        return conn;
    }
    
    /**
     * @brief Create coupling for axial motion only
     */
    static CylindricalCoupling axial_only(const BoundaryTarget& target,
                                           const Point<3>& axis_pt,
                                           const Tensor<1, 3>& axis_dir) {
        CylindricalCoupling conn;
        conn.target = target;
        conn.axis_point = axis_pt;
        conn.axis_direction = axis_dir / axis_dir.norm();
        conn.couple_axial = true;
        conn.description = "Axial cylindrical coupling";
        return conn;
    }
};

// ============================================================================
// Union Type for All Connections
// ============================================================================

/**
 * @brief Variant type holding any connection type
 */
using Connection = std::variant<
    SpringToGroundConnection,
    SpringConnection,
    BushingConnection,
    RigidConnection,
    TiedConnection,
    DirectionalCoupling,
    CylindricalCoupling
>;

/**
 * @brief Get description string from any connection type
 */
const std::string& get_connection_description(const Connection& conn);

/**
 * @brief Check if connection is a spring-type (contributes to stiffness matrix)
 */
bool is_spring_connection(const Connection& conn);

/**
 * @brief Check if connection is a constraint-type (contributes to AffineConstraints)
 */
bool is_constraint_connection(const Connection& conn);

// ============================================================================
// Connection Manager
// ============================================================================

/**
 * @brief Manages connection elements and applies them to the FE system
 * 
 * Handles:
 * - Spring stiffness matrix assembly
 * - Spring preload force assembly
 * - Constraint application for rigid/tied connections
 * - Post-processing of spring forces
 */
class ConnectionManager {
public:
    ConnectionManager();
    ~ConnectionManager() = default;
    
    // Connection management
    void add_connection(const Connection& conn);
    void clear();
    const std::vector<Connection>& get_connections() const { return connections_; }
    size_t size() const { return connections_.size(); }
    
    /**
     * @brief Apply constraint-type connections to AffineConstraints
     * 
     * Processes: RigidConnection, TiedConnection, DirectionalCoupling, CylindricalCoupling
     */
    template <int dim>
    void apply_to_constraints(
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    /**
     * @brief Assemble spring stiffness contributions to system matrix
     * 
     * Processes: SpringToGroundConnection, SpringConnection, BushingConnection
     */
    template <int dim>
    void assemble_spring_stiffness(
        SparseMatrix<double>& system_matrix,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    /**
     * @brief Assemble spring preload forces to RHS vector
     */
    template <int dim>
    void assemble_spring_preload(
        Vector<double>& system_rhs,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    /**
     * @brief Result structure for spring force calculation
     */
    template <int dim>
    struct SpringForceResult {
        std::string description;
        Point<dim> location_a;
        Point<dim> location_b;
        Tensor<1, dim> force;           ///< Force vector
        double axial_force;             ///< Axial component (positive = tension)
        double elongation;              ///< Spring elongation
    };
    
    /**
     * @brief Compute spring forces from displacement solution
     */
    template <int dim>
    std::vector<SpringForceResult<dim>> compute_spring_forces(
        const Vector<double>& solution,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    // Query methods
    bool has_spring_connections() const;
    bool has_constraint_connections() const;
    
    /**
     * @brief Get number of each connection type
     */
    std::map<std::string, size_t> get_connection_counts() const;
    
private:
    std::vector<Connection> connections_;
    
    /**
     * @brief Find DOF indices for a mesh node near a target point
     */
    template <int dim>
    std::array<types::global_dof_index, dim> find_node_dofs(
        const Point<dim>& target,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        double tolerance) const;
    
    /**
     * @brief Find all nodes on a boundary target
     */
    template <int dim>
    std::vector<std::pair<Point<dim>, std::array<types::global_dof_index, dim>>>
    find_boundary_nodes(
        const BoundaryTarget& target,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    // Specific connection handlers
    template <int dim>
    void apply_rigid_connection(
        const RigidConnection& conn,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_distributing_connection(
        const RigidConnection& conn,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_tied_connection(
        const TiedConnection& conn,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_directional_coupling(
        const DirectionalCoupling& conn,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void apply_cylindrical_coupling(
        const CylindricalCoupling& conn,
        AffineConstraints<double>& constraints,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_spring_to_ground(
        const SpringToGroundConnection& conn,
        SparseMatrix<double>& system_matrix,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_two_point_spring(
        const SpringConnection& conn,
        SparseMatrix<double>& system_matrix,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_bushing(
        const BushingConnection& conn,
        SparseMatrix<double>& system_matrix,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
};

} // namespace FEA

#endif // CONSTRAINT_BASE_H
