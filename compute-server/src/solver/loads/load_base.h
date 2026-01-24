/**
 * @file load_base.h
 * @brief Complete load system for FEA
 * 
 * Supports:
 * - Surface loads (force, pressure, hydrostatic)
 * - Point loads (force, moment, remote force, bearing)
 * - Body loads (gravity, acceleration, centrifugal)
 * - Thermal loads (uniform, field)
 */

#ifndef LOAD_BASE_H
#define LOAD_BASE_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>

#include "../boundary_conditions.h"  // For BoundaryTarget
#include "../material_library.h"

#include <variant>
#include <vector>
#include <functional>
#include <optional>
#include <string>
#include <map>
#include <memory>

namespace FEA {

using namespace dealii;

// ============================================================================
// SURFACE LOADS
// ============================================================================

/**
 * @brief Distributed traction (force per unit area) on a surface
 * 
 * Applies a force vector per unit area to a boundary surface.
 * Can be uniform or position-dependent.
 */
struct SurfaceForceLoad {
    BoundaryTarget target;
    
    // Constant force per unit area [N/m²] in global coordinates
    Tensor<1, 3> force_per_area;
    
    // For non-uniform loads: function of position
    // If set, overrides the constant value
    std::function<Tensor<1, 3>(const Point<3>&)> force_function;
    
    std::string description;
    
    SurfaceForceLoad() = default;
    SurfaceForceLoad(const BoundaryTarget& t, const Tensor<1, 3>& f, 
                     const std::string& desc = "")
        : target(t), force_per_area(f), description(desc) {}
    
    // Convenience constructors
    static SurfaceForceLoad uniform(const BoundaryTarget& target, 
                                    const Tensor<1, 3>& force);
    static SurfaceForceLoad uniform(const BoundaryTarget& target,
                                    double fx, double fy, double fz);
    static SurfaceForceLoad varying(const BoundaryTarget& target,
                                    std::function<Tensor<1, 3>(const Point<3>&)> func);
    
    // Get force at a point
    Tensor<1, 3> get_force_at(const Point<3>& p) const;
};

/**
 * @brief Pressure load (always normal to surface, positive = into surface)
 * 
 * Applies pressure normal to a boundary surface. Positive pressure
 * acts inward (compressive). For follower loads, the pressure direction
 * updates with the deformed surface normal.
 */
struct PressureLoad {
    BoundaryTarget target;
    double pressure;  // [Pa] positive = compressive (into surface)
    
    // Follower load: pressure direction follows deformed surface normal
    // Requires geometric nonlinearity
    bool is_follower = false;
    
    // For non-uniform pressure
    std::function<double(const Point<3>&)> pressure_function;
    
    std::string description;
    
    PressureLoad() = default;
    PressureLoad(const BoundaryTarget& t, double p, const std::string& desc = "")
        : target(t), pressure(p), description(desc) {}
    
    static PressureLoad uniform(const BoundaryTarget& target, double pressure);
    static PressureLoad follower(const BoundaryTarget& target, double pressure);
    static PressureLoad varying(const BoundaryTarget& target,
                               std::function<double(const Point<3>&)> func);
    
    double get_pressure_at(const Point<3>& p) const;
};

/**
 * @brief Hydrostatic pressure varying linearly with depth
 * 
 * Models fluid pressure that increases linearly with depth below
 * a free surface. Pressure is zero at the free surface.
 */
struct HydrostaticPressureLoad {
    BoundaryTarget target;
    double fluid_density;           // [kg/m³]
    double gravity_magnitude;       // [m/s²] typically 9.81
    Tensor<1, 3> gravity_direction; // Unit vector, typically [0,0,-1]
    Point<3> free_surface_point;    // Point where pressure = 0
    
    std::string description;
    
    HydrostaticPressureLoad() = default;
    HydrostaticPressureLoad(const BoundaryTarget& t, double rho, double g,
                            const Tensor<1, 3>& g_dir, const Point<3>& surface)
        : target(t), fluid_density(rho), gravity_magnitude(g),
          gravity_direction(g_dir / g_dir.norm()), free_surface_point(surface) {}
    
    // Compute pressure at a point
    double get_pressure_at(const Point<3>& p) const;
    
    static HydrostaticPressureLoad water(const BoundaryTarget& target,
                                         const Point<3>& free_surface);
};

// ============================================================================
// POINT LOADS
// ============================================================================

/**
 * @brief Concentrated force at a point
 * 
 * Applies a force at a specific location. For improved stress accuracy,
 * the force can be distributed over nodes within a radius.
 */
struct PointForceLoad {
    Point<3> location;
    Tensor<1, 3> force;  // [N]
    
    // If > 0, distribute force to nodes within radius using RBF weighting
    // This helps avoid stress singularities
    double distribution_radius = 0.0;
    
    std::string description;
    
    PointForceLoad() = default;
    PointForceLoad(const Point<3>& loc, const Tensor<1, 3>& f, 
                   const std::string& desc = "")
        : location(loc), force(f), description(desc) {}
    
    static PointForceLoad concentrated(const Point<3>& location, 
                                       const Tensor<1, 3>& force);
    static PointForceLoad concentrated(const Point<3>& location,
                                       double fx, double fy, double fz);
    static PointForceLoad distributed(const Point<3>& location,
                                      const Tensor<1, 3>& force,
                                      double radius);
};

/**
 * @brief Concentrated moment at a point
 * 
 * Since solid elements don't have rotational DOFs, moments are applied
 * either as force couples or through a rigid region.
 */
struct PointMomentLoad {
    Point<3> location;
    Tensor<1, 3> moment;  // [N·m]
    
    // How to apply the moment to a solid mesh (no rotational DOFs)
    enum class CouplingType {
        FORCE_COUPLE,   // Apply as pair of opposing forces
        RIGID_REGION    // Create rigid region and apply rotation
    };
    CouplingType coupling = CouplingType::FORCE_COUPLE;
    
    // Radius for force couple or rigid region
    double coupling_radius;
    
    std::string description;
    
    PointMomentLoad() = default;
    PointMomentLoad(const Point<3>& loc, const Tensor<1, 3>& m, double radius,
                    CouplingType type = CouplingType::FORCE_COUPLE)
        : location(loc), moment(m), coupling(type), coupling_radius(radius) {}
    
    static PointMomentLoad force_couple(const Point<3>& location,
                                        const Tensor<1, 3>& moment,
                                        double radius);
    static PointMomentLoad rigid_region(const Point<3>& location,
                                        const Tensor<1, 3>& moment,
                                        double radius);
};

/**
 * @brief Remote force: force applied at a point, coupled to a surface
 * 
 * Models loading through a rigid attachment (e.g., bolt head, lug).
 * The force and moment at the application point are distributed to
 * the target surface either rigidly or in a weighted manner.
 */
struct RemoteForceLoad {
    Point<3> application_point;
    BoundaryTarget target_surface;
    
    Tensor<1, 3> force;   // [N] at application point
    Tensor<1, 3> moment;  // [N·m] about application point
    
    // Coupling type
    enum class CouplingType {
        RIGID,       // All DOFs rigidly coupled (RBE2-like)
        DEFORMABLE   // Distributed coupling (RBE3-like)
    };
    CouplingType coupling = CouplingType::RIGID;
    
    std::string description;
    
    RemoteForceLoad() = default;
    
    static RemoteForceLoad rigid(const Point<3>& point,
                                 const BoundaryTarget& surface,
                                 const Tensor<1, 3>& force,
                                 const Tensor<1, 3>& moment = Tensor<1,3>());
    static RemoteForceLoad deformable(const Point<3>& point,
                                      const BoundaryTarget& surface,
                                      const Tensor<1, 3>& force);
};

/**
 * @brief Bearing load: cosine-distributed pressure on cylindrical hole
 * 
 * Models bolt bearing, pin contact, etc. The force is distributed
 * over the contact arc using a cosine distribution.
 */
struct BearingLoad {
    BoundaryTarget target;  // Should be a cylindrical surface
    Tensor<1, 3> force;     // Total force to distribute [N]
    Tensor<1, 3> cylinder_axis;
    Point<3> cylinder_center;
    double cylinder_radius;
    
    // Contact arc angle (default: 180° = π radians for half-cylinder contact)
    double contact_angle = M_PI;
    
    std::string description;
    
    BearingLoad() = default;
    BearingLoad(const BoundaryTarget& t, const Tensor<1, 3>& f,
                const Tensor<1, 3>& axis, const Point<3>& center, double radius)
        : target(t), force(f), cylinder_axis(axis / axis.norm()),
          cylinder_center(center), cylinder_radius(radius) {}
    
    // Get pressure at a point on the cylinder surface
    // Uses cosine distribution: p = p_max * max(0, cos(theta))
    // where theta is angle from force direction
    double get_pressure_at(const Point<3>& p, const Tensor<1, 3>& normal) const;
};

// ============================================================================
// BODY LOADS (Volume forces)
// ============================================================================

/**
 * @brief Gravitational body force
 * 
 * Applies acceleration * density as a body force throughout the volume.
 */
struct GravityLoad {
    Tensor<1, 3> acceleration;  // [m/s²] typically [0, 0, -9.81]
    
    // Limit to specific material regions (empty = all)
    std::vector<unsigned int> material_ids;
    
    std::string description;
    
    GravityLoad() = default;
    GravityLoad(const Tensor<1, 3>& g, const std::string& desc = "")
        : acceleration(g), description(desc) {}
    
    static GravityLoad standard();  // [0, 0, -9.81]
    static GravityLoad custom(double gx, double gy, double gz);
    static GravityLoad custom(const Tensor<1, 3>& accel);
};

/**
 * @brief Linear acceleration (for quasi-static inertia effects)
 * 
 * Similar to gravity but for modeling linear acceleration loads
 * like vehicle braking, elevator motion, etc.
 */
struct LinearAccelerationLoad {
    Tensor<1, 3> acceleration;  // [m/s²]
    std::vector<unsigned int> material_ids;
    std::string description;
    
    LinearAccelerationLoad() = default;
    LinearAccelerationLoad(const Tensor<1, 3>& a, const std::string& desc = "")
        : acceleration(a), description(desc) {}
    
    static LinearAccelerationLoad create(double ax, double ay, double az);
};

/**
 * @brief Centrifugal load (rotating body)
 * 
 * Models inertia loads from steady-state rotation about an axis.
 * Optionally includes tangential loads from angular acceleration.
 */
struct CentrifugalLoad {
    Point<3> axis_point;          // Point on rotation axis
    Tensor<1, 3> axis_direction;  // Unit vector along axis
    double angular_velocity;       // [rad/s]
    double angular_acceleration = 0.0;  // [rad/s²] for startup analysis
    
    std::vector<unsigned int> material_ids;
    std::string description;
    
    CentrifugalLoad() = default;
    CentrifugalLoad(const Point<3>& point, const Tensor<1, 3>& axis, double omega,
                    const std::string& desc = "")
        : axis_point(point), axis_direction(axis / axis.norm()), 
          angular_velocity(omega), description(desc) {}
    
    // Compute body force at a point: f = rho * omega² * r_vec
    // where r_vec is vector from axis to point, perpendicular to axis
    Tensor<1, 3> get_body_force(const Point<3>& p, double density) const;
    
    // Convert from RPM
    static CentrifugalLoad from_rpm(const Point<3>& axis_point,
                                    const Tensor<1, 3>& axis_direction,
                                    double rpm);
};

// ============================================================================
// THERMAL LOADS
// ============================================================================

/**
 * @brief Uniform temperature change throughout the model
 * 
 * Applies a constant temperature change from a reference (stress-free)
 * temperature, causing thermal strains ε = α * ΔT.
 */
struct UniformThermalLoad {
    double reference_temperature;  // Stress-free temperature [°C or K]
    double applied_temperature;    // Current temperature
    
    std::vector<unsigned int> material_ids;  // Empty = all
    std::string description;
    
    UniformThermalLoad() = default;
    UniformThermalLoad(double T_ref, double T_applied, const std::string& desc = "")
        : reference_temperature(T_ref), applied_temperature(T_applied),
          description(desc) {}
    
    double get_delta_T() const { 
        return applied_temperature - reference_temperature; 
    }
    
    static UniformThermalLoad heating(double T_ref, double T_final);
    static UniformThermalLoad cooling(double T_ref, double T_final);
};

/**
 * @brief Spatially varying temperature field
 * 
 * Temperature can vary throughout the model, specified either as
 * nodal values or as a function of position.
 */
struct TemperatureFieldLoad {
    double reference_temperature;
    
    // Option 1: nodal temperatures (indexed by global vertex index)
    std::map<unsigned int, double> nodal_temperatures;
    
    // Option 2: temperature as function of position
    std::function<double(const Point<3>&)> temperature_function;
    
    std::string description;
    
    TemperatureFieldLoad() : reference_temperature(0) {}
    TemperatureFieldLoad(double T_ref) : reference_temperature(T_ref) {}
    
    static TemperatureFieldLoad from_function(
        double T_ref, 
        std::function<double(const Point<3>&)> func,
        const std::string& desc = "");
    
    static TemperatureFieldLoad from_nodal_values(
        double T_ref,
        const std::map<unsigned int, double>& temperatures,
        const std::string& desc = "");
    
    // Linear gradient along an axis
    static TemperatureFieldLoad linear_gradient(
        double T_ref, double T_at_origin, const Tensor<1, 3>& gradient);
    
    double get_temperature_at(const Point<3>& p) const;
    double get_delta_T_at(const Point<3>& p) const;
};

// ============================================================================
// Union Type for All Loads
// ============================================================================

using Load = std::variant<
    SurfaceForceLoad,
    PressureLoad,
    HydrostaticPressureLoad,
    PointForceLoad,
    PointMomentLoad,
    RemoteForceLoad,
    BearingLoad,
    GravityLoad,
    LinearAccelerationLoad,
    CentrifugalLoad,
    UniformThermalLoad,
    TemperatureFieldLoad
>;

// Helper functions
const std::string& get_load_description(const Load& load);
bool is_surface_load(const Load& load);
bool is_point_load(const Load& load);
bool is_body_load(const Load& load);
bool is_thermal_load(const Load& load);
bool requires_nonlinear(const Load& load);  // e.g., follower pressure

// ============================================================================
// Load Manager Class
// ============================================================================

/**
 * @brief Manages and assembles all loads for the FE system
 * 
 * Handles collection of loads and their assembly into the RHS vector.
 * For nonlinear problems, can also compute load stiffness contributions.
 */
class LoadManager {
public:
    LoadManager();
    ~LoadManager() = default;
    
    // ===== Setup =====
    void add_load(const Load& load);
    void add_loads(const std::vector<Load>& loads);
    void remove_load(size_t index);
    void clear();
    
    const std::vector<Load>& get_loads() const { return loads_; }
    size_t size() const { return loads_.size(); }
    bool empty() const { return loads_.empty(); }
    
    // ===== Assembly =====
    
    /**
     * @brief Assemble all load contributions to the RHS vector
     */
    template <int dim>
    void assemble_rhs(
        Vector<double>& system_rhs,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const std::map<unsigned int, Material>& materials,
        const AffineConstraints<double>& constraints,
        const Vector<double>* current_solution = nullptr) const;
    
    /**
     * @brief For follower loads: get tangent stiffness contribution
     */
    template <int dim>
    void assemble_follower_stiffness(
        SparseMatrix<double>& system_matrix,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Vector<double>& current_solution) const;
    
    // ===== Query =====
    bool has_follower_loads() const;
    bool has_thermal_loads() const;
    bool has_body_loads() const;
    bool has_surface_loads() const;
    bool has_point_loads() const;
    
    // Get thermal strain at a point (for post-processing)
    SymmetricTensor<2, 3> get_thermal_strain(
        const Point<3>& p,
        const Material& material) const;
    
    // Get combined temperature delta at a point
    double get_temperature_delta(const Point<3>& p) const;
    
private:
    std::vector<Load> loads_;
    
    // ===== Individual Assembly Routines =====
    
    template <int dim>
    void assemble_surface_force(
        Vector<double>& rhs,
        const SurfaceForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_pressure(
        Vector<double>& rhs,
        const PressureLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const Vector<double>* current_solution) const;
    
    template <int dim>
    void assemble_hydrostatic_pressure(
        Vector<double>& rhs,
        const HydrostaticPressureLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_point_force(
        Vector<double>& rhs,
        const PointForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_point_moment(
        Vector<double>& rhs,
        const PointMomentLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_remote_force(
        Vector<double>& rhs,
        const RemoteForceLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_bearing(
        Vector<double>& rhs,
        const BearingLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping) const;
    
    template <int dim>
    void assemble_gravity(
        Vector<double>& rhs,
        const GravityLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const std::map<unsigned int, Material>& materials) const;
    
    template <int dim>
    void assemble_linear_acceleration(
        Vector<double>& rhs,
        const LinearAccelerationLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const std::map<unsigned int, Material>& materials) const;
    
    template <int dim>
    void assemble_centrifugal(
        Vector<double>& rhs,
        const CentrifugalLoad& load,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const std::map<unsigned int, Material>& materials) const;
    
    template <int dim>
    void assemble_thermal(
        Vector<double>& rhs,
        const DoFHandler<dim>& dof_handler,
        const Mapping<dim>& mapping,
        const std::map<unsigned int, Material>& materials) const;
};

} // namespace FEA

#endif // LOAD_BASE_H
