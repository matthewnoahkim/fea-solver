#ifndef ELASTIC_PROBLEM_H
#define ELASTIC_PROBLEM_H

/**
 * @file elastic_problem.h
 * @brief Main 3D static structural FEA solver class
 * 
 * This file defines the ElasticProblem class which orchestrates the complete
 * finite element analysis workflow:
 * - Mesh setup and refinement
 * - Material assignment
 * - Boundary condition application
 * - Load application
 * - Connection handling
 * - System assembly and solving
 * - Post-processing and result extraction
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include "material_library.h"
#include "boundary_conditions.h"
#include "loads/load_base.h"
#include "connections/constraint_base.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <optional>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

// ============================================================================
// Analysis Results Structures
// ============================================================================

/**
 * @brief Displacement field results
 */
struct DisplacementResults {
    double max_magnitude;               ///< Maximum displacement magnitude [user units]
    double max_x, max_y, max_z;         ///< Maximum displacement components
    double min_x, min_y, min_z;         ///< Minimum displacement components
    Point<3> max_magnitude_location;    ///< Location of max displacement
    
    /**
     * @brief Displacement at a sample point
     */
    struct SamplePointResult {
        Point<3> location;
        Tensor<1, 3> displacement;
    };
    std::vector<SamplePointResult> sample_points;
    
    json to_json() const;
};

/**
 * @brief Stress field results
 */
struct StressResults {
    double max_von_mises;               ///< Maximum von Mises stress [user units]
    Point<3> max_von_mises_location;    ///< Location of max von Mises
    
    std::array<double, 3> max_principal;    ///< Maximum principal stresses
    std::array<double, 3> min_principal;    ///< Minimum principal stresses
    
    double max_shear;                   ///< Maximum shear stress
    double max_tresca;                  ///< Maximum Tresca stress
    double max_equiv_plastic_strain;    ///< Max equivalent plastic strain (elastoplastic)
    
    /**
     * @brief Stress at a sample point
     */
    struct SamplePointResult {
        Point<3> location;
        double von_mises;
        std::array<double, 3> principal_stresses;
        SymmetricTensor<2, 3> stress_tensor;
    };
    std::vector<SamplePointResult> sample_points;
    
    json to_json() const;
};

/**
 * @brief Reaction force results
 */
struct ReactionResults {
    /**
     * @brief Reactions on a single boundary
     */
    struct BoundaryReaction {
        unsigned int boundary_id;
        std::string description;
        Tensor<1, 3> total_force;       ///< Total reaction force
        Tensor<1, 3> total_moment;      ///< Total reaction moment about centroid
        Point<3> centroid;              ///< Centroid of reaction nodes
    };
    std::vector<BoundaryReaction> boundary_reactions;
    
    Tensor<1, 3> total_force;           ///< Sum of all reaction forces
    Tensor<1, 3> total_moment;          ///< Sum of all reaction moments
    
    /**
     * @brief Equilibrium check results
     */
    struct EquilibriumCheck {
        double force_residual;          ///< |sum(forces)|
        double moment_residual;         ///< |sum(moments)|
        bool is_balanced;               ///< True if residuals < tolerance
    };
    EquilibriumCheck equilibrium;
    
    json to_json() const;
};

/**
 * @brief Strain energy results
 */
struct EnergyResults {
    double total_strain_energy;         ///< Total internal strain energy
    double max_strain_energy_density;   ///< Maximum strain energy density
    double total_external_work;         ///< Work done by external loads
    std::map<unsigned int, double> strain_energy_by_material;  ///< Per-material breakdown
    
    json to_json() const;
};

/**
 * @brief Safety factor results
 */
struct SafetyFactorResults {
    double min_yield_sf;                ///< Minimum yield safety factor
    Point<3> min_yield_sf_location;     ///< Location of min yield SF
    double min_ultimate_sf;             ///< Minimum ultimate safety factor
    Point<3> min_ultimate_sf_location;  ///< Location of min ultimate SF
    
    double percent_below_sf_1_0;        ///< % of volume with SF < 1.0
    double percent_below_sf_1_5;        ///< % of volume with SF < 1.5
    double percent_below_sf_2_0;        ///< % of volume with SF < 2.0
    
    /**
     * @brief Linearized stress result for ASME analysis
     */
    struct LinearizedStressResult {
        Point<3> start_point;
        Point<3> end_point;
        double membrane_stress;         ///< Pm (membrane)
        double bending_stress;          ///< Pb (bending)
        double peak_stress;             ///< F (peak)
        double membrane_plus_bending;   ///< Pm + Pb
    };
    std::vector<LinearizedStressResult> linearized_stresses;
    
    json to_json() const;
};

/**
 * @brief Mesh quality metrics
 */
struct MeshQualityResults {
    unsigned int num_elements;          ///< Number of active elements
    unsigned int num_nodes;             ///< Number of vertices
    unsigned int num_dofs;              ///< Number of degrees of freedom
    
    double min_jacobian_ratio;          ///< Minimum Jacobian determinant ratio
    double max_aspect_ratio;            ///< Maximum element aspect ratio
    double max_skewness;                ///< Maximum element skewness
    double max_warpage;                 ///< Maximum face warpage (3D)
    
    unsigned int num_poor_quality_elements;  ///< Elements failing quality checks
    bool quality_acceptable;            ///< Overall quality assessment
    
    json to_json() const;
};

/**
 * @brief Solver performance statistics
 */
struct SolverStatistics {
    unsigned int num_linear_iterations; ///< Linear solver iterations
    double final_residual;              ///< Final residual norm
    double computation_time_seconds;    ///< Total wall time
    
    unsigned int num_newton_iterations; ///< Newton-Raphson iterations (nonlinear)
    std::vector<double> newton_residuals;  ///< Residual history
    bool converged;                     ///< Convergence status
    
    json to_json() const;
};

/**
 * @brief Complete analysis results container
 */
struct AnalysisResults {
    DisplacementResults displacements;
    StressResults stresses;
    ReactionResults reactions;
    EnergyResults energy;
    SafetyFactorResults safety_factors;
    MeshQualityResults mesh_quality;
    SolverStatistics solver_stats;
    
    std::string vtk_output_path;        ///< Path to VTK output file
    std::string csv_output_path;        ///< Path to CSV output file
    
    json to_json() const;
};

// ============================================================================
// Solver Options
// ============================================================================

/**
 * @brief Configuration options for the FEA solver
 */
struct SolverOptions {
    // ===== Element Settings =====
    unsigned int fe_degree = 1;         ///< Polynomial degree (1=linear, 2=quadratic)
    
    // ===== Mesh Refinement =====
    unsigned int refinement_cycles = 0; ///< Initial global refinement cycles
    bool adaptive_refinement = false;   ///< Enable adaptive mesh refinement
    double adaptive_top_fraction = 0.3; ///< Fraction of cells to refine
    double adaptive_bottom_fraction = 0.03;  ///< Fraction of cells to coarsen
    unsigned int max_adaptive_cycles = 5;    ///< Maximum adaptive cycles
    
    // ===== Linear Solver =====
    unsigned int max_iterations = 10000;    ///< Maximum CG/GMRES iterations
    double tolerance = 1e-12;               ///< Convergence tolerance
    
    enum class SolverType { CG, GMRES, DIRECT };
    SolverType solver_type = SolverType::CG;
    
    enum class PreconditionerType { JACOBI, SSOR, AMG, ILU };
    PreconditionerType preconditioner = PreconditionerType::SSOR;
    
    // ===== Nonlinear Solver =====
    unsigned int max_newton_iterations = 20;
    double newton_tolerance = 1e-8;
    bool use_line_search = true;
    double line_search_alpha_min = 0.1;
    double line_search_reduction = 0.5;
    
    // ===== Analysis Type =====
    bool large_deformation = false;     ///< Enable geometric nonlinearity
    
    // ===== Output Options =====
    bool compute_stress = true;
    bool compute_strain = true;
    bool compute_principal_stress = true;
    bool compute_reactions = true;
    bool compute_safety_factors = true;
    bool output_vtk = true;
    bool output_csv = false;
    std::string output_directory = "./";
    
    // ===== Sample Points =====
    std::vector<Point<3>> sample_points;    ///< Points for detailed results
    
    // ===== Section Cuts (ASME Linearized Stress) =====
    std::vector<std::pair<Point<3>, Point<3>>> section_cuts;
    
    json to_json() const;
    static SolverOptions from_json(const json& j);
};

// ============================================================================
// Unit System
// ============================================================================

/**
 * @brief Unit system for input/output conversion
 * 
 * All internal calculations are performed in SI units (m, N, Pa).
 * Input/output values are converted based on the selected unit system.
 */
struct UnitSystem {
    enum class Type { 
        SI,             ///< m, N, Pa
        SI_MM,          ///< mm, N, MPa
        US_CUSTOMARY    ///< in, lbf, psi
    };
    Type type = Type::SI;
    
    double length_to_si;        ///< Multiply user length to get meters
    double force_to_si;         ///< Multiply user force to get Newtons
    double stress_to_si;        ///< Multiply user stress to get Pascals
    double stress_from_si;      ///< Multiply Pascals to get user stress
    
    /**
     * @brief SI unit system (m, N, Pa)
     */
    static UnitSystem SI() { 
        return {Type::SI, 1.0, 1.0, 1.0, 1.0}; 
    }
    
    /**
     * @brief SI-mm unit system (mm, N, MPa)
     */
    static UnitSystem SI_MM() { 
        return {Type::SI_MM, 1e-3, 1.0, 1e6, 1e-6}; 
    }
    
    /**
     * @brief US Customary unit system (in, lbf, psi)
     */
    static UnitSystem US_Customary() { 
        return {Type::US_CUSTOMARY, 0.0254, 4.44822, 6894.76, 1.0/6894.76}; 
    }
    
    /**
     * @brief Create from type enum
     */
    static UnitSystem from_type(Type t) {
        switch (t) {
            case Type::SI: return SI();
            case Type::SI_MM: return SI_MM();
            case Type::US_CUSTOMARY: return US_Customary();
            default: return SI();
        }
    }
    
    /**
     * @brief Parse from string
     */
    static UnitSystem from_string(const std::string& s) {
        if (s == "SI" || s == "si") return SI();
        if (s == "SI_MM" || s == "si_mm") return SI_MM();
        if (s == "US_CUSTOMARY" || s == "us_customary" || s == "imperial") 
            return US_Customary();
        return SI();
    }
};

// ============================================================================
// Main Solver Class
// ============================================================================

/**
 * @brief Main 3D static structural finite element solver
 * 
 * This class orchestrates the complete FEA workflow:
 * 1. Mesh setup (generation or import)
 * 2. Material assignment
 * 3. Boundary condition application
 * 4. Load application
 * 5. Connection handling (springs, rigid links, MPCs)
 * 6. System assembly
 * 7. Linear or nonlinear solving
 * 8. Post-processing
 * 9. Result output
 * 
 * @tparam dim Spatial dimension (default 3)
 * 
 * Example usage:
 * @code
 * ElasticProblem<3> problem;
 * problem.set_unit_system(UnitSystem::SI_MM());
 * problem.set_material_library(materials);
 * problem.read_mesh("model.msh");
 * problem.set_default_material("steel");
 * problem.add_boundary_condition(FixedBC::all(BoundaryTarget::from_boundary_id(1)));
 * problem.add_load(PressureLoad::uniform(BoundaryTarget::from_boundary_id(2), 10.0));
 * problem.run();
 * auto results = problem.get_results();
 * @endcode
 */
template <int dim = 3>
class ElasticProblem {
public:
    /**
     * @brief Construct solver with optional configuration
     */
    ElasticProblem(const SolverOptions& options = SolverOptions());
    
    /**
     * @brief Destructor
     */
    ~ElasticProblem();
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    /**
     * @brief Set the unit system for input/output
     */
    void set_unit_system(const UnitSystem& units);
    
    /**
     * @brief Update solver options
     */
    void set_options(const SolverOptions& options);
    
    // =========================================================================
    // Material Setup
    // =========================================================================
    
    /**
     * @brief Set the material library
     */
    void set_material_library(const MaterialLibrary& library);
    
    /**
     * @brief Assign a material to a mesh region
     * @param material_id The material_id from the mesh
     * @param material_name Name of material in library
     */
    void assign_material_to_region(unsigned int material_id, 
                                   const std::string& material_name);
    
    /**
     * @brief Set default material for unassigned regions
     */
    void set_default_material(const std::string& material_name);
    
    // =========================================================================
    // Boundary Conditions
    // =========================================================================
    
    /**
     * @brief Add a boundary condition
     */
    void add_boundary_condition(const BoundaryCondition& bc);
    
    /**
     * @brief Remove all boundary conditions
     */
    void clear_boundary_conditions();
    
    // =========================================================================
    // Loads
    // =========================================================================
    
    /**
     * @brief Add a load
     */
    void add_load(const Load& load);
    
    /**
     * @brief Remove all loads
     */
    void clear_loads();
    
    // =========================================================================
    // Connections
    // =========================================================================
    
    /**
     * @brief Add a connection (spring, rigid link, MPC, etc.)
     */
    void add_connection(const Connection& conn);
    
    /**
     * @brief Remove all connections
     */
    void clear_connections();
    
    // =========================================================================
    // Mesh Generation
    // =========================================================================
    
    /**
     * @brief Generate a box mesh
     * @param p1 First corner point
     * @param p2 Opposite corner point
     * @param subdivisions Number of elements in each direction
     */
    void generate_box_mesh(const Point<dim>& p1, const Point<dim>& p2,
                          const std::vector<unsigned int>& subdivisions);
    
    /**
     * @brief Generate a cylinder mesh
     * @param center Center point of cylinder
     * @param radius Cylinder radius
     * @param height Cylinder height (along z-axis)
     * @param n_radial Number of radial divisions
     * @param n_axial Number of axial divisions
     */
    void generate_cylinder_mesh(const Point<dim>& center, double radius,
                               double height, unsigned int n_radial,
                               unsigned int n_axial);
    
    /**
     * @brief Generate a sphere mesh
     * @param center Sphere center
     * @param radius Sphere radius
     * @param n_refinements Number of refinement cycles
     */
    void generate_sphere_mesh(const Point<dim>& center, double radius,
                             unsigned int n_refinements);
    
    // =========================================================================
    // Mesh Import
    // =========================================================================
    
    /**
     * @brief Read mesh from file
     * @param filename Path to mesh file (.msh, .vtk, .inp, .ucd)
     */
    void read_mesh(const std::string& filename);
    
    /**
     * @brief Read mesh from string data
     * @param data Mesh file contents
     * @param format Format identifier ("msh", "vtk", etc.)
     */
    void read_mesh_from_string(const std::string& data, const std::string& format);
    
    // =========================================================================
    // Mesh Refinement
    // =========================================================================
    
    /**
     * @brief Refine entire mesh globally
     */
    void refine_global(unsigned int times = 1);
    
    /**
     * @brief Refine mesh near a point
     */
    void refine_near_point(const Point<dim>& p, double radius, unsigned int times = 1);
    
    /**
     * @brief Refine mesh near a boundary
     */
    void refine_near_boundary(unsigned int boundary_id, unsigned int times = 1);
    
    // =========================================================================
    // Analysis Execution
    // =========================================================================
    
    /**
     * @brief Run the complete analysis
     * @throws std::runtime_error if mesh not loaded or no BCs defined
     */
    void run();
    
    /**
     * @brief Check if problem requires nonlinear solution
     */
    bool is_nonlinear() const;
    
    /**
     * @brief Get analysis results
     * @throws std::runtime_error if run() not called
     */
    const AnalysisResults& get_results() const;
    
    // =========================================================================
    // Output
    // =========================================================================
    
    /**
     * @brief Write results to VTK file for visualization
     */
    void output_vtk(const std::string& filename) const;
    
    /**
     * @brief Write results to CSV file
     */
    void output_csv(const std::string& filename) const;
    
    /**
     * @brief Get results as JSON
     */
    json results_to_json() const;
    
    // =========================================================================
    // Query Methods
    // =========================================================================
    
    /**
     * @brief Get number of degrees of freedom
     */
    unsigned int get_num_dofs() const { return dof_handler.n_dofs(); }
    
    /**
     * @brief Get number of active cells
     */
    unsigned int get_num_cells() const { return triangulation.n_active_cells(); }
    
    /**
     * @brief Get mesh quality metrics
     */
    MeshQualityResults get_mesh_quality() const;
    
    /**
     * @brief Get displacement at a point
     */
    Tensor<1, dim> get_displacement_at_point(const Point<dim>& p) const;
    
    /**
     * @brief Get stress tensor at a point
     */
    SymmetricTensor<2, dim> get_stress_at_point(const Point<dim>& p) const;
    
    // =========================================================================
    // Progress Tracking
    // =========================================================================
    
    /**
     * @brief Callback type for progress updates
     */
    using ProgressCallback = std::function<void(double progress, const std::string& stage)>;
    
    /**
     * @brief Set callback for progress updates
     */
    void set_progress_callback(ProgressCallback callback);
    
private:
    // =========================================================================
    // Setup Methods
    // =========================================================================
    
    void setup_system();
    void setup_fe();
    void setup_dofs();
    void setup_constraints();
    void setup_matrices();
    
    // =========================================================================
    // Assembly Methods
    // =========================================================================
    
    void assemble_system();
    void assemble_cell_matrix_rhs(
        const typename DoFHandler<dim>::active_cell_iterator& cell,
        FullMatrix<double>& cell_matrix,
        Vector<double>& cell_rhs,
        FEValues<dim>& fe_values) const;
    
    // =========================================================================
    // Solve Methods
    // =========================================================================
    
    void solve_linear();
    void solve_nonlinear();
    
    // =========================================================================
    // Post-Processing Methods
    // =========================================================================
    
    void compute_derived_quantities();
    void compute_stress_field();
    void compute_reactions();
    void compute_strain_energy();
    void compute_safety_factors();
    void compute_sample_point_results();
    void compute_linearized_stresses();
    void estimate_error();
    
    // =========================================================================
    // Helper Methods
    // =========================================================================
    
    /**
     * @brief Compute strain tensor at a quadrature point
     */
    SymmetricTensor<2, dim> get_strain(
        const FEValues<dim>& fe_values,
        const std::vector<types::global_dof_index>& dof_indices,
        unsigned int q_point) const;
    
    /**
     * @brief Compute stress from strain using material law
     */
    SymmetricTensor<2, dim> get_stress(
        const SymmetricTensor<2, dim>& strain,
        unsigned int material_id) const;
    
    /**
     * @brief Compute von Mises equivalent stress
     */
    double compute_von_mises(const SymmetricTensor<2, dim>& stress) const;
    
    /**
     * @brief Compute principal stresses
     */
    std::array<double, dim> compute_principal_stresses(
        const SymmetricTensor<2, dim>& stress) const;
    
    /**
     * @brief Get elasticity tensor for a material region
     */
    SymmetricTensor<4, dim> get_elasticity_tensor(unsigned int material_id) const;
    
    /**
     * @brief Report progress to callback
     */
    void report_progress(double progress, const std::string& stage);
    
    // =========================================================================
    // deal.II Objects
    // =========================================================================
    
    Triangulation<dim> triangulation;       ///< Mesh storage
    DoFHandler<dim> dof_handler;            ///< DOF distribution
    std::unique_ptr<FESystem<dim>> fe;      ///< Finite element
    MappingQ<dim> mapping;                  ///< Geometry mapping
    AffineConstraints<double> constraints;  ///< Constraint matrix
    
    SparsityPattern sparsity_pattern;       ///< Matrix sparsity
    SparseMatrix<double> system_matrix;     ///< Global stiffness matrix
    Vector<double> solution;                ///< Displacement solution
    Vector<double> system_rhs;              ///< Right-hand side vector
    
    // =========================================================================
    // Problem Data
    // =========================================================================
    
    SolverOptions options;                  ///< Solver configuration
    UnitSystem units;                       ///< Unit system
    MaterialLibrary material_library;       ///< Material database
    std::map<unsigned int, std::string> material_assignments;  ///< Region->material map
    std::string default_material;           ///< Default material name
    
    BoundaryConditionManager bc_manager;    ///< Boundary conditions
    LoadManager load_manager;               ///< Applied loads
    ConnectionManager connection_manager;   ///< Connections
    
    // =========================================================================
    // Results Storage
    // =========================================================================
    
    Vector<double> von_mises_field;         ///< Von Mises stress per cell
    std::vector<Vector<double>> principal_stress_fields;  ///< Principal stresses
    Vector<double> error_per_cell;          ///< Error estimator values
    AnalysisResults cached_results;         ///< Computed results
    bool results_valid;                     ///< Results validity flag
    
    // =========================================================================
    // Progress Tracking
    // =========================================================================
    
    ProgressCallback progress_callback;     ///< Progress callback
    Timer timer;                            ///< Timing
    
    // =========================================================================
    // State Flags
    // =========================================================================
    
    bool system_setup_done;                 ///< System setup completed
    bool mesh_loaded;                       ///< Mesh loaded/generated
};

} // namespace FEA

#endif // ELASTIC_PROBLEM_H
