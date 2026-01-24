#include "elastic_problem.h"
#include <deal.II/base/work_stream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace FEA {

// ============================================================================
// JSON Serialization for Result Structures
// ============================================================================

json DisplacementResults::to_json() const {
    json j;
    j["max_magnitude"] = max_magnitude;
    j["max_x"] = max_x;
    j["max_y"] = max_y;
    j["max_z"] = max_z;
    j["min_x"] = min_x;
    j["min_y"] = min_y;
    j["min_z"] = min_z;
    j["max_magnitude_location"] = {
        max_magnitude_location[0],
        max_magnitude_location[1],
        max_magnitude_location[2]
    };
    
    json samples = json::array();
    for (const auto& sp : sample_points) {
        samples.push_back({
            {"location", {sp.location[0], sp.location[1], sp.location[2]}},
            {"displacement", {sp.displacement[0], sp.displacement[1], sp.displacement[2]}}
        });
    }
    j["sample_points"] = samples;
    
    return j;
}

json StressResults::to_json() const {
    json j;
    j["max_von_mises"] = max_von_mises;
    j["max_von_mises_location"] = {
        max_von_mises_location[0],
        max_von_mises_location[1],
        max_von_mises_location[2]
    };
    j["max_principal"] = max_principal;
    j["min_principal"] = min_principal;
    j["max_shear"] = max_shear;
    j["max_tresca"] = max_tresca;
    j["max_equiv_plastic_strain"] = max_equiv_plastic_strain;
    
    json samples = json::array();
    for (const auto& sp : sample_points) {
        samples.push_back({
            {"location", {sp.location[0], sp.location[1], sp.location[2]}},
            {"von_mises", sp.von_mises},
            {"principal_stresses", sp.principal_stresses}
        });
    }
    j["sample_points"] = samples;
    
    return j;
}

json ReactionResults::to_json() const {
    json j;
    j["total_force"] = {total_force[0], total_force[1], total_force[2]};
    j["total_moment"] = {total_moment[0], total_moment[1], total_moment[2]};
    
    json boundaries = json::array();
    for (const auto& br : boundary_reactions) {
        boundaries.push_back({
            {"boundary_id", br.boundary_id},
            {"description", br.description},
            {"total_force", {br.total_force[0], br.total_force[1], br.total_force[2]}},
            {"total_moment", {br.total_moment[0], br.total_moment[1], br.total_moment[2]}},
            {"centroid", {br.centroid[0], br.centroid[1], br.centroid[2]}}
        });
    }
    j["boundaries"] = boundaries;
    
    j["equilibrium"] = {
        {"force_residual", equilibrium.force_residual},
        {"moment_residual", equilibrium.moment_residual},
        {"is_balanced", equilibrium.is_balanced}
    };
    
    return j;
}

json EnergyResults::to_json() const {
    json j;
    j["total_strain_energy"] = total_strain_energy;
    j["max_strain_energy_density"] = max_strain_energy_density;
    j["total_external_work"] = total_external_work;
    j["strain_energy_by_material"] = strain_energy_by_material;
    return j;
}

json SafetyFactorResults::to_json() const {
    json j;
    j["min_yield_sf"] = min_yield_sf;
    j["min_yield_sf_location"] = {
        min_yield_sf_location[0],
        min_yield_sf_location[1],
        min_yield_sf_location[2]
    };
    j["min_ultimate_sf"] = min_ultimate_sf;
    j["percent_below_sf_1_0"] = percent_below_sf_1_0;
    j["percent_below_sf_1_5"] = percent_below_sf_1_5;
    j["percent_below_sf_2_0"] = percent_below_sf_2_0;
    
    json linearized = json::array();
    for (const auto& ls : linearized_stresses) {
        linearized.push_back({
            {"start_point", {ls.start_point[0], ls.start_point[1], ls.start_point[2]}},
            {"end_point", {ls.end_point[0], ls.end_point[1], ls.end_point[2]}},
            {"membrane_stress", ls.membrane_stress},
            {"bending_stress", ls.bending_stress},
            {"peak_stress", ls.peak_stress},
            {"membrane_plus_bending", ls.membrane_plus_bending}
        });
    }
    j["linearized_stresses"] = linearized;
    
    return j;
}

json MeshQualityResults::to_json() const {
    return {
        {"num_elements", num_elements},
        {"num_nodes", num_nodes},
        {"num_dofs", num_dofs},
        {"min_jacobian_ratio", min_jacobian_ratio},
        {"max_aspect_ratio", max_aspect_ratio},
        {"max_skewness", max_skewness},
        {"max_warpage", max_warpage},
        {"num_poor_quality_elements", num_poor_quality_elements},
        {"quality_acceptable", quality_acceptable}
    };
}

json SolverStatistics::to_json() const {
    return {
        {"num_linear_iterations", num_linear_iterations},
        {"final_residual", final_residual},
        {"computation_time_seconds", computation_time_seconds},
        {"num_newton_iterations", num_newton_iterations},
        {"newton_residuals", newton_residuals},
        {"converged", converged}
    };
}

json AnalysisResults::to_json() const {
    return {
        {"summary", {
            {"num_dofs", mesh_quality.num_dofs},
            {"num_elements", mesh_quality.num_elements},
            {"solver_iterations", solver_stats.num_linear_iterations},
            {"computation_time_seconds", solver_stats.computation_time_seconds},
            {"converged", solver_stats.converged}
        }},
        {"displacements", displacements.to_json()},
        {"stresses", stresses.to_json()},
        {"reactions", reactions.to_json()},
        {"energy", energy.to_json()},
        {"safety_factors", safety_factors.to_json()},
        {"mesh_quality", mesh_quality.to_json()},
        {"solver_stats", solver_stats.to_json()},
        {"vtk_output_path", vtk_output_path},
        {"csv_output_path", csv_output_path}
    };
}

json SolverOptions::to_json() const {
    return {
        {"fe_degree", fe_degree},
        {"refinement_cycles", refinement_cycles},
        {"adaptive_refinement", adaptive_refinement},
        {"max_iterations", max_iterations},
        {"tolerance", tolerance},
        {"solver_type", static_cast<int>(solver_type)},
        {"large_deformation", large_deformation},
        {"compute_stress", compute_stress},
        {"compute_reactions", compute_reactions},
        {"output_vtk", output_vtk}
    };
}

SolverOptions SolverOptions::from_json(const json& j) {
    SolverOptions opts;
    
    if (j.contains("fe_degree")) opts.fe_degree = j["fe_degree"];
    if (j.contains("refinement_cycles")) opts.refinement_cycles = j["refinement_cycles"];
    if (j.contains("adaptive_refinement")) opts.adaptive_refinement = j["adaptive_refinement"];
    if (j.contains("max_iterations")) opts.max_iterations = j["max_iterations"];
    if (j.contains("tolerance")) opts.tolerance = j["tolerance"];
    if (j.contains("large_deformation")) opts.large_deformation = j["large_deformation"];
    if (j.contains("compute_stress")) opts.compute_stress = j["compute_stress"];
    if (j.contains("compute_reactions")) opts.compute_reactions = j["compute_reactions"];
    if (j.contains("output_vtk")) opts.output_vtk = j["output_vtk"];
    
    if (j.contains("solver_type")) {
        std::string st = j["solver_type"];
        if (st == "CG" || st == "cg") opts.solver_type = SolverType::CG;
        else if (st == "GMRES" || st == "gmres") opts.solver_type = SolverType::GMRES;
        else if (st == "DIRECT" || st == "direct") opts.solver_type = SolverType::DIRECT;
    }
    
    if (j.contains("sample_points")) {
        for (const auto& sp : j["sample_points"]) {
            opts.sample_points.push_back(Point<3>(sp[0], sp[1], sp[2]));
        }
    }
    
    return opts;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

template <int dim>
ElasticProblem<dim>::ElasticProblem(const SolverOptions& opts)
    : dof_handler(triangulation)
    , mapping(1)
    , options(opts)
    , units(UnitSystem::SI())
    , results_valid(false)
    , system_setup_done(false)
    , mesh_loaded(false)
{
    setup_fe();
}

template <int dim>
ElasticProblem<dim>::~ElasticProblem() {
    dof_handler.clear();
}

// ============================================================================
// Configuration
// ============================================================================

template <int dim>
void ElasticProblem<dim>::set_unit_system(const UnitSystem& u) {
    units = u;
}

template <int dim>
void ElasticProblem<dim>::set_options(const SolverOptions& opts) {
    // Check if FE degree changed
    bool fe_changed = (opts.fe_degree != options.fe_degree);
    
    options = opts;
    
    if (fe_changed) {
        setup_fe();
        system_setup_done = false;
    }
}

template <int dim>
void ElasticProblem<dim>::set_material_library(const MaterialLibrary& library) {
    material_library = library;
}

template <int dim>
void ElasticProblem<dim>::assign_material_to_region(unsigned int material_id,
                                                     const std::string& material_name) {
    if (!material_library.has_material(material_name))
        throw std::runtime_error("Material not found in library: " + material_name);
    material_assignments[material_id] = material_name;
}

template <int dim>
void ElasticProblem<dim>::set_default_material(const std::string& material_name) {
    if (!material_library.has_material(material_name))
        throw std::runtime_error("Material not found in library: " + material_name);
    default_material = material_name;
}

// ============================================================================
// Boundary Conditions, Loads, Connections
// ============================================================================

template <int dim>
void ElasticProblem<dim>::add_boundary_condition(const BoundaryCondition& bc) {
    bc_manager.add_condition(bc);
}

template <int dim>
void ElasticProblem<dim>::clear_boundary_conditions() {
    bc_manager.clear();
}

template <int dim>
void ElasticProblem<dim>::add_load(const Load& load) {
    load_manager.add_load(load);
}

template <int dim>
void ElasticProblem<dim>::clear_loads() {
    load_manager.clear();
}

template <int dim>
void ElasticProblem<dim>::add_connection(const Connection& conn) {
    connection_manager.add_connection(conn);
}

template <int dim>
void ElasticProblem<dim>::clear_connections() {
    connection_manager.clear();
}

// ============================================================================
// Mesh Generation
// ============================================================================

template <int dim>
void ElasticProblem<dim>::generate_box_mesh(
    const Point<dim>& p1, const Point<dim>& p2,
    const std::vector<unsigned int>& subdivisions) {
    
    triangulation.clear();
    
    // Convert from user units to SI
    Point<dim> p1_si, p2_si;
    for (unsigned int d = 0; d < dim; ++d) {
        p1_si[d] = p1[d] * units.length_to_si;
        p2_si[d] = p2[d] * units.length_to_si;
    }
    
    // Generate mesh
    GridGenerator::subdivided_hyper_rectangle(
        triangulation, subdivisions, p1_si, p2_si, true);
    
    // The 'true' flag automatically assigns boundary IDs:
    // 0: x = p1[0], 1: x = p2[0]
    // 2: y = p1[1], 3: y = p2[1]
    // 4: z = p1[2], 5: z = p2[2]
    
    mesh_loaded = true;
    system_setup_done = false;
    results_valid = false;
    
    report_progress(0.05, "Box mesh generated: " + 
        std::to_string(triangulation.n_active_cells()) + " cells");
}

template <int dim>
void ElasticProblem<dim>::generate_cylinder_mesh(
    const Point<dim>& center, double radius, double height,
    unsigned int n_radial, unsigned int n_axial) {
    
    triangulation.clear();
    
    // Scale to SI
    double radius_si = radius * units.length_to_si;
    double height_si = height * units.length_to_si;
    Point<dim> center_si;
    for (unsigned int d = 0; d < dim; ++d)
        center_si[d] = center[d] * units.length_to_si;
    
    // Generate cylinder mesh along z-axis
    if constexpr (dim == 3) {
        GridGenerator::cylinder(triangulation, radius_si, height_si / 2.0);
        
        // Rotate to align with z-axis (deal.II generates along x-axis)
        GridTools::rotate(numbers::PI / 2.0, 1, triangulation);
    } else {
        // 2D: create a rectangle for the cross-section
        GridGenerator::hyper_rectangle(triangulation,
            Point<dim>(-radius_si, 0),
            Point<dim>(radius_si, height_si),
            true);
    }
    
    // Move to specified center
    GridTools::shift(center_si, triangulation);
    
    // Refine to get desired resolution
    unsigned int refinements = static_cast<unsigned int>(
        std::log2(std::max(n_radial, n_axial)) - 1);
    refinements = std::max(1u, refinements);
    triangulation.refine_global(refinements);
    
    mesh_loaded = true;
    system_setup_done = false;
    results_valid = false;
    
    report_progress(0.05, "Cylinder mesh generated");
}

template <int dim>
void ElasticProblem<dim>::generate_sphere_mesh(
    const Point<dim>& center, double radius, unsigned int n_refinements) {
    
    triangulation.clear();
    
    double radius_si = radius * units.length_to_si;
    Point<dim> center_si;
    for (unsigned int d = 0; d < dim; ++d)
        center_si[d] = center[d] * units.length_to_si;
    
    GridGenerator::hyper_ball(triangulation, center_si, radius_si);
    triangulation.refine_global(n_refinements);
    
    mesh_loaded = true;
    system_setup_done = false;
    results_valid = false;
    
    report_progress(0.05, "Sphere mesh generated");
}

// ============================================================================
// Mesh Import
// ============================================================================

template <int dim>
void ElasticProblem<dim>::read_mesh(const std::string& filename) {
    triangulation.clear();
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    
    std::ifstream input_file(filename);
    if (!input_file)
        throw std::runtime_error("Cannot open mesh file: " + filename);
    
    // Determine format from extension
    std::string extension;
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos)
        extension = filename.substr(dot_pos + 1);
    
    // Convert to lowercase
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == "msh") {
        grid_in.read_msh(input_file);
    } else if (extension == "vtk") {
        grid_in.read_vtk(input_file);
    } else if (extension == "inp") {
        grid_in.read_abaqus(input_file);
    } else if (extension == "ucd") {
        grid_in.read_ucd(input_file);
    } else if (extension == "exo" || extension == "e") {
        // Exodus format - would need ExodusII library
        throw std::runtime_error("Exodus format requires ExodusII library");
    } else {
        throw std::runtime_error("Unsupported mesh format: " + extension);
    }
    
    // Scale mesh if not in SI units
    if (std::abs(units.length_to_si - 1.0) > 1e-12) {
        GridTools::scale(units.length_to_si, triangulation);
    }
    
    mesh_loaded = true;
    system_setup_done = false;
    results_valid = false;
    
    report_progress(0.05, "Mesh loaded: " + 
        std::to_string(triangulation.n_active_cells()) + " cells, " +
        std::to_string(triangulation.n_vertices()) + " vertices");
}

template <int dim>
void ElasticProblem<dim>::read_mesh_from_string(
    const std::string& data, const std::string& format) {
    
    triangulation.clear();
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    
    std::istringstream input_stream(data);
    
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
    
    if (fmt == "msh" || fmt == "gmsh") {
        grid_in.read_msh(input_stream);
    } else if (fmt == "vtk") {
        grid_in.read_vtk(input_stream);
    } else if (fmt == "ucd") {
        grid_in.read_ucd(input_stream);
    } else {
        throw std::runtime_error("Unsupported mesh format for string input: " + format);
    }
    
    if (std::abs(units.length_to_si - 1.0) > 1e-12) {
        GridTools::scale(units.length_to_si, triangulation);
    }
    
    mesh_loaded = true;
    system_setup_done = false;
    results_valid = false;
}

// ============================================================================
// Mesh Refinement
// ============================================================================

template <int dim>
void ElasticProblem<dim>::refine_global(unsigned int times) {
    triangulation.refine_global(times);
    system_setup_done = false;
    results_valid = false;
}

template <int dim>
void ElasticProblem<dim>::refine_near_point(
    const Point<dim>& p, double radius, unsigned int times) {
    
    Point<dim> p_si;
    for (unsigned int d = 0; d < dim; ++d)
        p_si[d] = p[d] * units.length_to_si;
    double radius_si = radius * units.length_to_si;
    
    for (unsigned int cycle = 0; cycle < times; ++cycle) {
        for (auto& cell : triangulation.active_cell_iterators()) {
            if (cell->center().distance(p_si) < radius_si)
                cell->set_refine_flag();
        }
        triangulation.execute_coarsening_and_refinement();
    }
    
    system_setup_done = false;
    results_valid = false;
}

template <int dim>
void ElasticProblem<dim>::refine_near_boundary(
    unsigned int boundary_id, unsigned int times) {
    
    for (unsigned int cycle = 0; cycle < times; ++cycle) {
        for (auto& cell : triangulation.active_cell_iterators()) {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
                if (cell->face(f)->at_boundary() &&
                    cell->face(f)->boundary_id() == boundary_id) {
                    cell->set_refine_flag();
                    break;
                }
            }
        }
        triangulation.execute_coarsening_and_refinement();
    }
    
    system_setup_done = false;
    results_valid = false;
}

// ============================================================================
// Setup Methods
// ============================================================================

template <int dim>
void ElasticProblem<dim>::setup_fe() {
    fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(options.fe_degree), dim);
}

template <int dim>
void ElasticProblem<dim>::setup_system() {
    report_progress(0.1, "Setting up system");
    
    setup_dofs();
    setup_constraints();
    setup_matrices();
    
    system_setup_done = true;
}

template <int dim>
void ElasticProblem<dim>::setup_dofs() {
    dof_handler.distribute_dofs(*fe);
    
    report_progress(0.12, "DOFs distributed: " + std::to_string(dof_handler.n_dofs()));
}

template <int dim>
void ElasticProblem<dim>::setup_constraints() {
    constraints.clear();
    
    // Hanging node constraints (for adaptive refinement)
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    
    // Apply boundary conditions
    bc_manager.apply_to_constraints(constraints, dof_handler, mapping);
    
    // Apply connection constraints (rigid links, MPCs)
    if (connection_manager.has_constraint_connections()) {
        connection_manager.apply_to_constraints(constraints, dof_handler, mapping);
    }
    
    constraints.close();
    
    report_progress(0.15, "Constraints applied: " + 
        std::to_string(constraints.n_constraints()) + " constraint lines");
}

template <int dim>
void ElasticProblem<dim>::setup_matrices() {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    
    sparsity_pattern.copy_from(dsp);
    
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    
    report_progress(0.18, "Matrices allocated: " + 
        std::to_string(sparsity_pattern.n_nonzero_elements()) + " nonzeros");
}

// ============================================================================
// Assembly
// ============================================================================

template <int dim>
void ElasticProblem<dim>::assemble_system() {
    report_progress(0.2, "Assembling system");
    
    system_matrix = 0;
    system_rhs = 0;
    
    QGauss<dim> quadrature(fe->degree + 1);
    QGauss<dim-1> face_quadrature(fe->degree + 1);
    
    FEValues<dim> fe_values(*fe, quadrature,
        update_values | update_gradients | 
        update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();
    
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    // Precompute elasticity tensors for each material region
    std::map<unsigned int, SymmetricTensor<4, dim>> material_tensors;
    for (const auto& [region_id, mat_name] : material_assignments) {
        material_tensors[region_id] = get_elasticity_tensor(region_id);
    }
    
    // Default material tensor
    SymmetricTensor<4, dim> default_C;
    if (!default_material.empty()) {
        const Material& mat = material_library.get_material(default_material);
        if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat.properties)) {
            default_C = iso->get_elasticity_tensor();
        }
    }
    
    unsigned int cell_count = 0;
    const unsigned int total_cells = triangulation.n_active_cells();
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
        
        // Get material elasticity tensor for this cell
        unsigned int mat_id = cell->material_id();
        SymmetricTensor<4, dim> C;
        
        auto it = material_tensors.find(mat_id);
        if (it != material_tensors.end()) {
            C = it->second;
        } else if (!default_material.empty()) {
            C = default_C;
        } else {
            throw std::runtime_error("No material assigned to region " + 
                                    std::to_string(mat_id) + 
                                    " and no default material set");
        }
        
        // Assemble stiffness matrix: K_ij = integral(B_i^T * C * B_j)
        for (unsigned int q = 0; q < n_q_points; ++q) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const unsigned int comp_i = fe->system_to_component_index(i).first;
                
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const unsigned int comp_j = fe->system_to_component_index(j).first;
                    
                    // Compute B_i^T * C * B_j contribution
                    double contrib = 0;
                    for (unsigned int d1 = 0; d1 < dim; ++d1) {
                        for (unsigned int d2 = 0; d2 < dim; ++d2) {
                            contrib += fe_values.shape_grad(i, q)[d1] *
                                      C[comp_i][d1][comp_j][d2] *
                                      fe_values.shape_grad(j, q)[d2];
                        }
                    }
                    
                    cell_matrix(i, j) += contrib * fe_values.JxW(q);
                }
            }
        }
        
        // Get local DOF indices
        cell->get_dof_indices(local_dof_indices);
        
        // Distribute to global system with constraints
        constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices,
            system_matrix, system_rhs);
        
        ++cell_count;
        if (cell_count % 1000 == 0) {
            double progress = 0.2 + 0.35 * (static_cast<double>(cell_count) / total_cells);
            report_progress(progress, "Assembling cells: " + 
                std::to_string(cell_count) + "/" + std::to_string(total_cells));
        }
    }
    
    report_progress(0.55, "Cell assembly complete");
    
    // Add elastic support stiffness contributions
    if (bc_manager.has_elastic_support_conditions()) {
        auto elastic_entries = bc_manager.get_elastic_support_matrix_entries(
            dof_handler, mapping, face_quadrature);
        
        for (const auto& [row, col, value] : elastic_entries) {
            system_matrix.add(row, col, value);
        }
        report_progress(0.57, "Elastic supports added");
    }
    
    // Add spring stiffness contributions
    if (connection_manager.has_spring_connections()) {
        connection_manager.assemble_spring_stiffness(system_matrix, dof_handler, mapping);
        connection_manager.assemble_spring_preload(system_rhs, dof_handler, mapping);
        report_progress(0.58, "Spring connections added");
    }
    
    // Assemble loads into RHS
    // Note: LoadManager needs material info for thermal loads
    std::map<unsigned int, Material> cell_materials;
    for (const auto& [region_id, mat_name] : material_assignments) {
        cell_materials[region_id] = material_library.get_material(mat_name);
    }
    
    load_manager.assemble_rhs(system_rhs, dof_handler, mapping,
                              cell_materials, constraints);
    
    report_progress(0.6, "Assembly complete");
}

// ============================================================================
// Solve
// ============================================================================

template <int dim>
void ElasticProblem<dim>::solve_linear() {
    report_progress(0.65, "Solving linear system");
    
    Timer solve_timer;
    solve_timer.start();
    
    if (options.solver_type == SolverOptions::SolverType::DIRECT) {
        // Direct solver (UMFPACK)
        SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(system_matrix);
        direct_solver.vmult(solution, system_rhs);
        
        cached_results.solver_stats.num_linear_iterations = 1;
        cached_results.solver_stats.final_residual = 0;
    }
    else if (options.solver_type == SolverOptions::SolverType::CG) {
        // Conjugate Gradient with SSOR preconditioner
        SolverControl solver_control(options.max_iterations, 
                                     options.tolerance * system_rhs.l2_norm());
        SolverCG<Vector<double>> solver(solver_control);
        
        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);
        
        solver.solve(system_matrix, solution, system_rhs, preconditioner);
        
        cached_results.solver_stats.num_linear_iterations = solver_control.last_step();
        cached_results.solver_stats.final_residual = solver_control.last_value();
    }
    else {
        // GMRES with Jacobi preconditioner
        SolverControl solver_control(options.max_iterations, 
                                     options.tolerance * system_rhs.l2_norm());
        SolverGMRES<Vector<double>> solver(solver_control);
        
        PreconditionJacobi<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix);
        
        solver.solve(system_matrix, solution, system_rhs, preconditioner);
        
        cached_results.solver_stats.num_linear_iterations = solver_control.last_step();
        cached_results.solver_stats.final_residual = solver_control.last_value();
    }
    
    // Distribute constrained DOF values
    constraints.distribute(solution);
    
    solve_timer.stop();
    cached_results.solver_stats.computation_time_seconds = solve_timer.wall_time();
    cached_results.solver_stats.converged = true;
    
    report_progress(0.75, "Solve complete: " + 
        std::to_string(cached_results.solver_stats.num_linear_iterations) + 
        " iterations, residual = " + 
        std::to_string(cached_results.solver_stats.final_residual));
}

template <int dim>
void ElasticProblem<dim>::solve_nonlinear() {
    // Newton-Raphson iteration for nonlinear problems
    // (elastoplastic materials, large deformation, contact)
    
    report_progress(0.65, "Starting nonlinear solve (Newton-Raphson)");
    
    Vector<double> residual(dof_handler.n_dofs());
    Vector<double> delta_u(dof_handler.n_dofs());
    
    cached_results.solver_stats.newton_residuals.clear();
    
    // Initial guess (could use previous solution if available)
    solution = 0;
    
    for (unsigned int newton_iter = 0; newton_iter < options.max_newton_iterations; 
         ++newton_iter) {
        
        // Assemble tangent stiffness and residual at current solution
        // Note: For true nonlinear, assembly would account for current deformation
        assemble_system();
        
        // Compute residual: R = F_ext - F_int = RHS - K*u
        residual = system_rhs;
        system_matrix.vmult_add(residual, solution);
        residual *= -1.0;
        // Now residual = K*u - RHS
        
        // Actually for equilibrium: K*u = F
        // Residual for Newton: R(u) = F - K(u)*u for linear, or F_int(u) - F_ext
        // Let's use: R = F_ext - K*u_current
        residual = system_rhs;
        Vector<double> temp(dof_handler.n_dofs());
        system_matrix.vmult(temp, solution);
        residual -= temp;
        
        double residual_norm = residual.l2_norm();
        cached_results.solver_stats.newton_residuals.push_back(residual_norm);
        
        report_progress(0.65 + 0.05 * newton_iter, 
            "Newton iteration " + std::to_string(newton_iter + 1) + 
            ", residual = " + std::to_string(residual_norm));
        
        // Check convergence
        if (residual_norm < options.newton_tolerance) {
            cached_results.solver_stats.num_newton_iterations = newton_iter + 1;
            cached_results.solver_stats.converged = true;
            constraints.distribute(solution);
            report_progress(0.75, "Newton converged in " + 
                std::to_string(newton_iter + 1) + " iterations");
            return;
        }
        
        // Solve for increment: K * delta_u = R
        SolverControl solver_control(options.max_iterations, 
                                     options.tolerance * residual_norm);
        SolverCG<Vector<double>> solver(solver_control);
        
        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);
        
        delta_u = 0;
        solver.solve(system_matrix, delta_u, residual, preconditioner);
        
        // Line search (optional backtracking)
        double alpha = 1.0;
        if (options.use_line_search) {
            // Simple backtracking line search
            Vector<double> trial_solution = solution;
            Vector<double> trial_residual(dof_handler.n_dofs());
            
            double initial_norm = residual_norm;
            
            for (unsigned int ls_iter = 0; ls_iter < 10; ++ls_iter) {
                trial_solution = solution;
                trial_solution.add(alpha, delta_u);
                
                // Would need to reassemble to get true residual
                // Simplified: use linear approximation
                system_matrix.vmult(trial_residual, trial_solution);
                trial_residual -= system_rhs;
                
                if (trial_residual.l2_norm() < initial_norm ||
                    alpha < options.line_search_alpha_min) {
                    break;
                }
                
                alpha *= options.line_search_reduction;
            }
        }
        
        // Update solution
        solution.add(alpha, delta_u);
        constraints.distribute(solution);
    }
    
    // Failed to converge
    cached_results.solver_stats.num_newton_iterations = options.max_newton_iterations;
    cached_results.solver_stats.converged = false;
    throw std::runtime_error("Newton-Raphson did not converge within " +
        std::to_string(options.max_newton_iterations) + " iterations");
}

// ============================================================================
// Main Run Method
// ============================================================================

template <int dim>
void ElasticProblem<dim>::run() {
    timer.restart();
    
    // Validate setup
    if (!mesh_loaded)
        throw std::runtime_error("No mesh loaded. Call read_mesh() or generate_*_mesh() first.");
    
    if (bc_manager.size() == 0)
        throw std::runtime_error("No boundary conditions defined. Model is unconstrained.");
    
    if (default_material.empty() && material_assignments.empty())
        throw std::runtime_error("No materials assigned. Call set_default_material() or assign_material_to_region().");
    
    report_progress(0.0, "Starting analysis");
    
    // Initial global refinement
    if (options.refinement_cycles > 0) {
        refine_global(options.refinement_cycles);
        report_progress(0.02, "Initial refinement complete");
    }
    
    // Adaptive refinement loop
    unsigned int cycle = 0;
    do {
        // Setup system if needed
        if (!system_setup_done)
            setup_system();
        
        // Assemble
        assemble_system();
        
        // Solve
        if (is_nonlinear())
            solve_nonlinear();
        else
            solve_linear();
        
        // Adaptive refinement
        if (options.adaptive_refinement && cycle < options.max_adaptive_cycles) {
            estimate_error();
            
            GridRefinement::refine_and_coarsen_fixed_number(
                triangulation, error_per_cell,
                options.adaptive_top_fraction,
                options.adaptive_bottom_fraction);
            
            triangulation.execute_coarsening_and_refinement();
            system_setup_done = false;
            
            report_progress(0.75 + 0.02 * cycle, 
                "Adaptive refinement cycle " + std::to_string(cycle + 1));
        }
        
        ++cycle;
    } while (options.adaptive_refinement && 
             cycle <= options.max_adaptive_cycles && 
             !system_setup_done);
    
    // Post-processing
    compute_derived_quantities();
    
    // Store mesh quality results
    cached_results.mesh_quality = get_mesh_quality();
    
    // Output files
    if (options.output_vtk) {
        std::string vtk_filename = options.output_directory + "results.vtu";
        output_vtk(vtk_filename);
        cached_results.vtk_output_path = vtk_filename;
    }
    
    if (options.output_csv) {
        std::string csv_filename = options.output_directory + "results.csv";
        output_csv(csv_filename);
        cached_results.csv_output_path = csv_filename;
    }
    
    results_valid = true;
    
    timer.stop();
    cached_results.solver_stats.computation_time_seconds = timer.wall_time();
    
    report_progress(1.0, "Analysis complete in " + 
        std::to_string(timer.wall_time()) + " seconds");
}

template <int dim>
bool ElasticProblem<dim>::is_nonlinear() const {
    // Check if any material is nonlinear
    for (const auto& [region_id, mat_name] : material_assignments) {
        const Material& mat = material_library.get_material(mat_name);
        if (mat.is_nonlinear())
            return true;
    }
    
    if (!default_material.empty()) {
        const Material& mat = material_library.get_material(default_material);
        if (mat.is_nonlinear())
            return true;
    }
    
    // Check for large deformation
    if (options.large_deformation)
        return true;
    
    // Check for follower loads
    if (load_manager.has_follower_loads())
        return true;
    
    // Check for contact
    if (bc_manager.has_contact_conditions())
        return true;
    
    return false;
}

// ============================================================================
// Post-Processing
// ============================================================================

template <int dim>
void ElasticProblem<dim>::compute_derived_quantities() {
    report_progress(0.8, "Computing derived quantities");
    
    if (options.compute_stress)
        compute_stress_field();
    
    if (options.compute_reactions)
        compute_reactions();
    
    compute_strain_energy();
    
    if (options.compute_safety_factors)
        compute_safety_factors();
    
    if (!options.sample_points.empty())
        compute_sample_point_results();
    
    if (!options.section_cuts.empty())
        compute_linearized_stresses();
    
    // Compute displacement extrema
    double max_disp_mag = 0;
    double max_x = -std::numeric_limits<double>::max();
    double min_x = std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_z = -std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    Point<dim> max_disp_loc;
    
    std::vector<bool> vertex_visited(triangulation.n_vertices(), false);
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);
        
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            unsigned int vertex_index = cell->vertex_index(v);
            if (vertex_visited[vertex_index]) continue;
            vertex_visited[vertex_index] = true;
            
            Point<dim> vertex = cell->vertex(v);
            Tensor<1, dim> disp;
            
            // Get displacement components at this vertex
            for (unsigned int d = 0; d < dim; ++d) {
                unsigned int local_dof = fe->component_to_system_index(d, v);
                disp[d] = solution(local_dof_indices[local_dof]);
            }
            
            // Update extrema (in user units)
            double disp_x = disp[0] / units.length_to_si;
            double disp_y = disp[1] / units.length_to_si;
            double disp_z = (dim == 3) ? disp[2] / units.length_to_si : 0.0;
            
            max_x = std::max(max_x, disp_x);
            min_x = std::min(min_x, disp_x);
            max_y = std::max(max_y, disp_y);
            min_y = std::min(min_y, disp_y);
            if (dim == 3) {
                max_z = std::max(max_z, disp_z);
                min_z = std::min(min_z, disp_z);
            }
            
            double mag = disp.norm() / units.length_to_si;
            if (mag > max_disp_mag) {
                max_disp_mag = mag;
                max_disp_loc = vertex;
            }
        }
    }
    
    // Store displacement results
    cached_results.displacements.max_magnitude = max_disp_mag;
    cached_results.displacements.max_x = max_x;
    cached_results.displacements.max_y = max_y;
    cached_results.displacements.max_z = max_z;
    cached_results.displacements.min_x = min_x;
    cached_results.displacements.min_y = min_y;
    cached_results.displacements.min_z = min_z;
    
    // Convert location to user units
    for (unsigned int d = 0; d < dim; ++d)
        cached_results.displacements.max_magnitude_location[d] = 
            max_disp_loc[d] / units.length_to_si;
    
    report_progress(0.95, "Post-processing complete");
}

template <int dim>
void ElasticProblem<dim>::compute_stress_field() {
    QGauss<dim> quadrature(fe->degree + 1);
    FEValues<dim> fe_values(*fe, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    von_mises_field.reinit(triangulation.n_active_cells());
    
    double max_vm = 0;
    Point<dim> max_vm_loc;
    std::array<double, dim> max_principal = {-std::numeric_limits<double>::max(),
                                              -std::numeric_limits<double>::max(),
                                              -std::numeric_limits<double>::max()};
    std::array<double, dim> min_principal = {std::numeric_limits<double>::max(),
                                              std::numeric_limits<double>::max(),
                                              std::numeric_limits<double>::max()};
    double max_shear = 0;
    
    unsigned int cell_index = 0;
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        
        std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);
        
        double cell_max_vm = 0;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            // Compute strain tensor
            SymmetricTensor<2, dim> strain = get_strain(fe_values, local_dof_indices, q);
            
            // Compute stress
            SymmetricTensor<2, dim> stress = get_stress(strain, cell->material_id());
            
            // Von Mises stress
            double vm = compute_von_mises(stress);
            cell_max_vm = std::max(cell_max_vm, vm);
            
            if (vm > max_vm) {
                max_vm = vm;
                max_vm_loc = fe_values.quadrature_point(q);
            }
            
            // Principal stresses
            if (options.compute_principal_stress) {
                auto principals = compute_principal_stresses(stress);
                for (unsigned int d = 0; d < dim; ++d) {
                    max_principal[d] = std::max(max_principal[d], principals[d]);
                    min_principal[d] = std::min(min_principal[d], principals[d]);
                }
                
                // Max shear = (sigma_1 - sigma_3) / 2
                double shear = (principals[0] - principals[dim-1]) / 2.0;
                max_shear = std::max(max_shear, std::abs(shear));
            }
        }
        
        von_mises_field(cell_index) = cell_max_vm;
        ++cell_index;
    }
    
    // Store stress results (converted to user units)
    cached_results.stresses.max_von_mises = max_vm * units.stress_from_si;
    for (unsigned int d = 0; d < dim; ++d)
        cached_results.stresses.max_von_mises_location[d] = max_vm_loc[d] / units.length_to_si;
    
    for (unsigned int d = 0; d < dim; ++d) {
        cached_results.stresses.max_principal[d] = max_principal[d] * units.stress_from_si;
        cached_results.stresses.min_principal[d] = min_principal[d] * units.stress_from_si;
    }
    cached_results.stresses.max_shear = max_shear * units.stress_from_si;
    cached_results.stresses.max_tresca = 2.0 * max_shear * units.stress_from_si;
}

template <int dim>
void ElasticProblem<dim>::compute_reactions() {
    // Compute reaction forces at constrained DOFs
    // Reactions = K * u - F at constrained nodes
    
    Vector<double> full_rhs(dof_handler.n_dofs());
    
    // Recompute RHS without constraint distribution to get applied loads
    full_rhs = 0;
    load_manager.assemble_rhs(full_rhs, dof_handler, mapping,
        std::map<unsigned int, Material>(), AffineConstraints<double>());
    
    // Compute K*u
    Vector<double> Ku(dof_handler.n_dofs());
    system_matrix.vmult(Ku, solution);
    
    // Reactions = K*u - F
    Vector<double> reactions = Ku;
    reactions -= full_rhs;
    
    // Sum reactions by boundary
    cached_results.reactions.total_force = Tensor<1, dim>();
    cached_results.reactions.total_moment = Tensor<1, dim>();
    cached_results.reactions.boundary_reactions.clear();
    
    // Get boundaries with Dirichlet conditions
    auto dirichlet_ids = bc_manager.get_dirichlet_boundary_ids();
    
    for (auto boundary_id : dirichlet_ids) {
        Tensor<1, dim> total_force;
        Point<dim> centroid;
        unsigned int num_nodes = 0;
        
        std::set<types::global_dof_index> counted_dofs;
        
        for (const auto& cell : dof_handler.active_cell_iterators()) {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
                if (!cell->face(f)->at_boundary()) continue;
                if (cell->face(f)->boundary_id() != boundary_id) continue;
                
                std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
                cell->get_dof_indices(local_dof_indices);
                
                for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                    unsigned int cell_v = GeometryInfo<dim>::face_to_cell_vertices(f, v);
                    Point<dim> vertex = cell->vertex(cell_v);
                    
                    for (unsigned int d = 0; d < dim; ++d) {
                        types::global_dof_index dof = 
                            local_dof_indices[fe->component_to_system_index(d, cell_v)];
                        
                        if (counted_dofs.count(dof) == 0 && 
                            constraints.is_constrained(dof)) {
                            counted_dofs.insert(dof);
                            total_force[d] += reactions(dof);
                        }
                    }
                    
                    centroid += vertex;
                    ++num_nodes;
                }
            }
        }
        
        if (num_nodes > 0) {
            centroid /= num_nodes;
            
            ReactionResults::BoundaryReaction br;
            br.boundary_id = boundary_id;
            br.description = "Boundary " + std::to_string(boundary_id);
            
            for (unsigned int d = 0; d < dim; ++d) {
                br.total_force[d] = total_force[d] * units.force_to_si;
                br.centroid[d] = centroid[d] / units.length_to_si;
            }
            
            // Compute moment about centroid (placeholder - would need proper implementation)
            br.total_moment = Tensor<1, dim>();
            
            cached_results.reactions.boundary_reactions.push_back(br);
            cached_results.reactions.total_force += br.total_force;
        }
    }
    
    // Equilibrium check
    double force_residual = cached_results.reactions.total_force.norm();
    cached_results.reactions.equilibrium.force_residual = force_residual;
    cached_results.reactions.equilibrium.is_balanced = (force_residual < 1e-6);
}

template <int dim>
void ElasticProblem<dim>::compute_strain_energy() {
    QGauss<dim> quadrature(fe->degree + 1);
    FEValues<dim> fe_values(*fe, quadrature,
        update_values | update_gradients | update_JxW_values);
    
    double total_energy = 0;
    double max_energy_density = 0;
    std::map<unsigned int, double> energy_by_material;
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        
        std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);
        
        unsigned int mat_id = cell->material_id();
        double cell_energy = 0;
        double cell_volume = 0;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            SymmetricTensor<2, dim> strain = get_strain(fe_values, local_dof_indices, q);
            SymmetricTensor<2, dim> stress = get_stress(strain, mat_id);
            
            // Strain energy density = 0.5 * sigma : epsilon
            double energy_density = 0.5 * (stress * strain);
            
            cell_energy += energy_density * fe_values.JxW(q);
            cell_volume += fe_values.JxW(q);
        }
        
        total_energy += cell_energy;
        energy_by_material[mat_id] += cell_energy;
        
        if (cell_volume > 0) {
            double avg_density = cell_energy / cell_volume;
            max_energy_density = std::max(max_energy_density, avg_density);
        }
    }
    
    cached_results.energy.total_strain_energy = total_energy;
    cached_results.energy.max_strain_energy_density = max_energy_density;
    cached_results.energy.strain_energy_by_material = energy_by_material;
    
    // External work = F · u
    cached_results.energy.total_external_work = system_rhs * solution;
}

template <int dim>
void ElasticProblem<dim>::compute_safety_factors() {
    // Compute safety factors against yield and ultimate
    // SF_yield = sigma_yield / sigma_von_mises
    
    double min_yield_sf = std::numeric_limits<double>::max();
    Point<dim> min_yield_loc;
    
    QGauss<dim> quadrature(fe->degree + 1);
    FEValues<dim> fe_values(*fe, quadrature,
        update_values | update_gradients | update_quadrature_points);
    
    double vol_below_1_0 = 0, vol_below_1_5 = 0, vol_below_2_0 = 0;
    double total_volume = 0;
    
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        
        std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);
        
        // Get yield stress for this material
        unsigned int mat_id = cell->material_id();
        double yield_stress = 0;
        
        std::string mat_name = default_material;
        if (material_assignments.count(mat_id))
            mat_name = material_assignments.at(mat_id);
        
        if (!mat_name.empty()) {
            const Material& mat = material_library.get_material(mat_name);
            if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat.properties)) {
                yield_stress = iso->yield_strength;
            } else if (auto* ep = std::get_if<ElastoplasticVonMisesProperties>(&mat.properties)) {
                yield_stress = ep->initial_yield_stress;
            }
        }
        
        if (yield_stress <= 0) continue;  // Skip if no yield data
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            SymmetricTensor<2, dim> strain = get_strain(fe_values, local_dof_indices, q);
            SymmetricTensor<2, dim> stress = get_stress(strain, mat_id);
            double vm = compute_von_mises(stress);
            
            double sf = (vm > 1e-12) ? yield_stress / vm : 
                        std::numeric_limits<double>::max();
            
            if (sf < min_yield_sf) {
                min_yield_sf = sf;
                min_yield_loc = fe_values.quadrature_point(q);
            }
            
            double dV = fe_values.JxW(q);
            total_volume += dV;
            if (sf < 1.0) vol_below_1_0 += dV;
            if (sf < 1.5) vol_below_1_5 += dV;
            if (sf < 2.0) vol_below_2_0 += dV;
        }
    }
    
    cached_results.safety_factors.min_yield_sf = min_yield_sf;
    for (unsigned int d = 0; d < dim; ++d)
        cached_results.safety_factors.min_yield_sf_location[d] = 
            min_yield_loc[d] / units.length_to_si;
    
    if (total_volume > 0) {
        cached_results.safety_factors.percent_below_sf_1_0 = 100.0 * vol_below_1_0 / total_volume;
        cached_results.safety_factors.percent_below_sf_1_5 = 100.0 * vol_below_1_5 / total_volume;
        cached_results.safety_factors.percent_below_sf_2_0 = 100.0 * vol_below_2_0 / total_volume;
    }
}

template <int dim>
void ElasticProblem<dim>::compute_sample_point_results() {
    for (const auto& sample_pt : options.sample_points) {
        Point<dim> pt_si;
        for (unsigned int d = 0; d < dim; ++d)
            pt_si[d] = sample_pt[d] * units.length_to_si;
        
        try {
            Tensor<1, dim> disp = get_displacement_at_point(pt_si);
            
            DisplacementResults::SamplePointResult dr;
            for (unsigned int d = 0; d < dim; ++d) {
                dr.location[d] = sample_pt[d];
                dr.displacement[d] = disp[d] / units.length_to_si;
            }
            cached_results.displacements.sample_points.push_back(dr);
            
            SymmetricTensor<2, dim> stress = get_stress_at_point(pt_si);
            double vm = compute_von_mises(stress);
            auto principals = compute_principal_stresses(stress);
            
            StressResults::SamplePointResult sr;
            for (unsigned int d = 0; d < dim; ++d)
                sr.location[d] = sample_pt[d];
            sr.von_mises = vm * units.stress_from_si;
            for (unsigned int d = 0; d < dim; ++d)
                sr.principal_stresses[d] = principals[d] * units.stress_from_si;
            sr.stress_tensor = stress;
            cached_results.stresses.sample_points.push_back(sr);
        } catch (...) {
            // Point outside mesh - skip
        }
    }
}

template <int dim>
void ElasticProblem<dim>::compute_linearized_stresses() {
    // ASME linearized stress along section cuts
    // Placeholder implementation
    for (const auto& [start, end] : options.section_cuts) {
        SafetyFactorResults::LinearizedStressResult ls;
        ls.start_point = start;
        ls.end_point = end;
        // Would need proper integration along the section line
        ls.membrane_stress = 0;
        ls.bending_stress = 0;
        ls.peak_stress = 0;
        ls.membrane_plus_bending = 0;
        cached_results.safety_factors.linearized_stresses.push_back(ls);
    }
}

template <int dim>
void ElasticProblem<dim>::estimate_error() {
    // Kelly error estimator
    error_per_cell.reinit(triangulation.n_active_cells());
    
    KellyErrorEstimator<dim>::estimate(
        dof_handler, 
        QGauss<dim-1>(fe->degree + 1),
        std::map<types::boundary_id, const Function<dim>*>(),
        solution,
        error_per_cell);
}

// ============================================================================
// Helper Methods
// ============================================================================

template <int dim>
SymmetricTensor<2, dim> ElasticProblem<dim>::get_strain(
    const FEValues<dim>& fe_values,
    const std::vector<types::global_dof_index>& dof_indices,
    unsigned int q_point) const {
    
    SymmetricTensor<2, dim> strain;
    
    for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i) {
        const unsigned int comp = fe->system_to_component_index(i).first;
        const double u_i = solution(dof_indices[i]);
        
        for (unsigned int d = 0; d < dim; ++d) {
            // epsilon_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
            strain[comp][d] += 0.5 * u_i * fe_values.shape_grad(i, q_point)[d];
            strain[d][comp] += 0.5 * u_i * fe_values.shape_grad(i, q_point)[d];
        }
    }
    
    return strain;
}

template <int dim>
SymmetricTensor<2, dim> ElasticProblem<dim>::get_stress(
    const SymmetricTensor<2, dim>& strain, unsigned int material_id) const {
    
    SymmetricTensor<4, dim> C = get_elasticity_tensor(material_id);
    return C * strain;
}

template <int dim>
double ElasticProblem<dim>::compute_von_mises(
    const SymmetricTensor<2, dim>& stress) const {
    
    // von Mises = sqrt(3 * J2)
    // J2 = 0.5 * s_ij * s_ij where s = deviatoric stress
    
    double trace = 0;
    for (unsigned int d = 0; d < dim; ++d)
        trace += stress[d][d];
    
    double mean = trace / 3.0;  // Use 3 for 3D von Mises even in 2D
    
    SymmetricTensor<2, dim> deviatoric = stress;
    for (unsigned int d = 0; d < dim; ++d)
        deviatoric[d][d] -= mean;
    
    double J2 = 0.5 * (deviatoric * deviatoric);
    return std::sqrt(3.0 * J2);
}

template <int dim>
std::array<double, dim> ElasticProblem<dim>::compute_principal_stresses(
    const SymmetricTensor<2, dim>& stress) const {
    
    auto eigenvalues = eigenvalues_symmetric(stress);
    
    std::array<double, dim> principals;
    for (unsigned int d = 0; d < dim; ++d)
        principals[d] = eigenvalues[d];
    
    // Sort descending (sigma_1 >= sigma_2 >= sigma_3)
    std::sort(principals.begin(), principals.end(), std::greater<double>());
    
    return principals;
}

template <int dim>
SymmetricTensor<4, dim> ElasticProblem<dim>::get_elasticity_tensor(
    unsigned int material_id) const {
    
    std::string mat_name;
    if (material_assignments.count(material_id))
        mat_name = material_assignments.at(material_id);
    else
        mat_name = default_material;
    
    if (mat_name.empty())
        throw std::runtime_error("No material for region " + std::to_string(material_id));
    
    const Material& mat = material_library.get_material(mat_name);
    
    if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat.properties)) {
        return iso->get_elasticity_tensor();
    }
    else if (auto* ortho = std::get_if<OrthotropicElasticProperties>(&mat.properties)) {
        return ortho->get_elasticity_tensor();
    }
    else if (auto* ep = std::get_if<ElastoplasticVonMisesProperties>(&mat.properties)) {
        return ep->get_elastic_tensor();
    }
    
    throw std::runtime_error("Unsupported material type for elasticity tensor");
}

template <int dim>
Tensor<1, dim> ElasticProblem<dim>::get_displacement_at_point(
    const Point<dim>& p) const {
    
    // Find cell containing point
    auto cell_and_point = GridTools::find_active_cell_around_point(
        mapping, dof_handler, p);
    
    auto cell = cell_and_point.first;
    Point<dim> ref_point = cell_and_point.second;
    
    // Evaluate solution at point
    Quadrature<dim> point_quadrature(ref_point);
    FEValues<dim> fe_values(*fe, point_quadrature, update_values);
    fe_values.reinit(cell);
    
    std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
    cell->get_dof_indices(local_dof_indices);
    
    Tensor<1, dim> displacement;
    for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i) {
        const unsigned int comp = fe->system_to_component_index(i).first;
        displacement[comp] += solution(local_dof_indices[i]) * 
                              fe_values.shape_value(i, 0);
    }
    
    return displacement;
}

template <int dim>
SymmetricTensor<2, dim> ElasticProblem<dim>::get_stress_at_point(
    const Point<dim>& p) const {
    
    auto cell_and_point = GridTools::find_active_cell_around_point(
        mapping, dof_handler, p);
    
    auto cell = cell_and_point.first;
    Point<dim> ref_point = cell_and_point.second;
    
    Quadrature<dim> point_quadrature(ref_point);
    FEValues<dim> fe_values(*fe, point_quadrature, 
        update_values | update_gradients);
    fe_values.reinit(cell);
    
    std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
    cell->get_dof_indices(local_dof_indices);
    
    SymmetricTensor<2, dim> strain = get_strain(fe_values, local_dof_indices, 0);
    return get_stress(strain, cell->material_id());
}

template <int dim>
MeshQualityResults ElasticProblem<dim>::get_mesh_quality() const {
    MeshQualityResults mq;
    
    mq.num_elements = triangulation.n_active_cells();
    mq.num_nodes = triangulation.n_vertices();
    mq.num_dofs = dof_handler.n_dofs();
    
    // Compute quality metrics
    mq.min_jacobian_ratio = std::numeric_limits<double>::max();
    mq.max_aspect_ratio = 0;
    mq.max_skewness = 0;
    mq.num_poor_quality_elements = 0;
    
    QGauss<dim> quadrature(2);
    FEValues<dim> fe_values(*fe, quadrature, update_jacobians);
    
    for (const auto& cell : triangulation.active_cell_iterators()) {
        fe_values.reinit(cell);
        
        // Jacobian quality
        double min_J = std::numeric_limits<double>::max();
        double max_J = 0;
        
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            double J = fe_values.jacobian(q).determinant();
            min_J = std::min(min_J, J);
            max_J = std::max(max_J, std::abs(J));
        }
        
        if (max_J > 1e-14) {
            double ratio = min_J / max_J;
            mq.min_jacobian_ratio = std::min(mq.min_jacobian_ratio, ratio);
            
            if (ratio < 0.1) {
                mq.num_poor_quality_elements++;
            }
        }
        
        // Aspect ratio (simplified)
        double min_edge = std::numeric_limits<double>::max();
        double max_edge = 0;
        
        for (unsigned int e = 0; e < GeometryInfo<dim>::lines_per_cell; ++e) {
            double edge_len = cell->line(e)->diameter();
            min_edge = std::min(min_edge, edge_len);
            max_edge = std::max(max_edge, edge_len);
        }
        
        if (min_edge > 1e-14) {
            double ar = max_edge / min_edge;
            mq.max_aspect_ratio = std::max(mq.max_aspect_ratio, ar);
        }
    }
    
    mq.quality_acceptable = (mq.min_jacobian_ratio > 0.1 && 
                             mq.max_aspect_ratio < 100 &&
                             mq.num_poor_quality_elements == 0);
    
    return mq;
}

template <int dim>
void ElasticProblem<dim>::report_progress(double progress, const std::string& stage) {
    if (progress_callback)
        progress_callback(progress, stage);
}

template <int dim>
const AnalysisResults& ElasticProblem<dim>::get_results() const {
    if (!results_valid)
        throw std::runtime_error("No valid results available. Call run() first.");
    return cached_results;
}

template <int dim>
void ElasticProblem<dim>::set_progress_callback(ProgressCallback callback) {
    progress_callback = callback;
}

template <int dim>
json ElasticProblem<dim>::results_to_json() const {
    return get_results().to_json();
}

// ============================================================================
// Output Methods
// ============================================================================

template <int dim>
void ElasticProblem<dim>::output_vtk(const std::string& filename) const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    
    // Displacement field
    std::vector<std::string> solution_names(dim);
    solution_names[0] = "displacement_x";
    solution_names[1] = "displacement_y";
    if constexpr (dim == 3) solution_names[2] = "displacement_z";
    
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    
    data_out.add_data_vector(solution, solution_names,
        DataOut<dim>::type_dof_data, interpretation);
    
    // Von Mises stress (cell data)
    if (von_mises_field.size() == triangulation.n_active_cells()) {
        data_out.add_data_vector(von_mises_field, "von_mises_stress",
            DataOut<dim>::type_cell_data);
    }
    
    // Material IDs
    Vector<double> material_ids(triangulation.n_active_cells());
    unsigned int idx = 0;
    for (const auto& cell : triangulation.active_cell_iterators()) {
        material_ids(idx++) = cell->material_id();
    }
    data_out.add_data_vector(material_ids, "material_id",
        DataOut<dim>::type_cell_data);
    
    data_out.build_patches();
    
    std::ofstream output(filename);
    data_out.write_vtu(output);
}

template <int dim>
void ElasticProblem<dim>::output_csv(const std::string& filename) const {
    std::ofstream output(filename);
    
    output << "x,y,z,disp_x,disp_y,disp_z,von_mises\n";
    
    QGauss<dim> quadrature(1);  // Cell centers
    FEValues<dim> fe_values(*fe, quadrature,
        update_values | update_gradients | update_quadrature_points);
    
    unsigned int cell_idx = 0;
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        
        std::vector<types::global_dof_index> local_dof_indices(fe->n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);
        
        Point<dim> center = fe_values.quadrature_point(0);
        
        Tensor<1, dim> disp;
        for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i) {
            const unsigned int comp = fe->system_to_component_index(i).first;
            disp[comp] += solution(local_dof_indices[i]) * fe_values.shape_value(i, 0);
        }
        
        double vm = (von_mises_field.size() > cell_idx) ? 
                    von_mises_field(cell_idx) : 0.0;
        
        output << std::setprecision(8)
               << center[0] / units.length_to_si << ","
               << center[1] / units.length_to_si << ","
               << (dim == 3 ? center[2] / units.length_to_si : 0.0) << ","
               << disp[0] / units.length_to_si << ","
               << disp[1] / units.length_to_si << ","
               << (dim == 3 ? disp[2] / units.length_to_si : 0.0) << ","
               << vm * units.stress_from_si << "\n";
        
        ++cell_idx;
    }
}

// ============================================================================
// Explicit Template Instantiation
// ============================================================================

template class ElasticProblem<3>;
template class ElasticProblem<2>;

} // namespace FEA
