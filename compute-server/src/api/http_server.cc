#include "http_server.h"
#include "../mesh/mesh_quality.h"
#include "../mesh/mesh_reader.h"
#include "../post/stress_calculator.h"

#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace FEA {

// ============================================================================
// Job Status Conversion
// ============================================================================

std::string to_string(JobStatus status) {
    switch (status) {
        case JobStatus::QUEUED: return "queued";
        case JobStatus::RUNNING: return "running";
        case JobStatus::COMPLETED: return "completed";
        case JobStatus::FAILED: return "failed";
        case JobStatus::CANCELLED: return "cancelled";
    }
    return "unknown";
}

JobStatus status_from_string(const std::string& str) {
    if (str == "queued") return JobStatus::QUEUED;
    if (str == "running") return JobStatus::RUNNING;
    if (str == "completed") return JobStatus::COMPLETED;
    if (str == "failed") return JobStatus::FAILED;
    if (str == "cancelled") return JobStatus::CANCELLED;
    return JobStatus::QUEUED;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

FEAServer::FEAServer(int p, int nw, const std::string& dd)
    : port(p)
    , data_dir(dd)
    , running(false)
    , num_workers(nw) {
    
    // Create data directories
    std::filesystem::create_directories(data_dir + "/jobs");
    std::filesystem::create_directories(data_dir + "/meshes");
    std::filesystem::create_directories(data_dir + "/temp");
}

FEAServer::~FEAServer() {
    stop();
}

// ============================================================================
// Server Lifecycle
// ============================================================================

void FEAServer::start() {
    running = true;
    start_time = std::chrono::system_clock::now();
    
    // Start worker threads
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(&FEAServer::worker_thread, this);
    }
    
    // =========================================================================
    // Setup Routes
    // =========================================================================
    
    // Analysis submission
    server.Post("/api/analyze",
        [this](const auto& req, auto& res) { handle_analyze(req, res); });
    
    // Job status
    server.Get(R"(/api/jobs/([a-zA-Z0-9_-]+))",
        [this](const auto& req, auto& res) { handle_job_status(req, res); });
    
    // Job results
    server.Get(R"(/api/jobs/([a-zA-Z0-9_-]+)/results)",
        [this](const auto& req, auto& res) { handle_job_results(req, res); });
    
    // Job files (VTK, CSV, etc.)
    server.Get(R"(/api/jobs/([a-zA-Z0-9_-]+)/files/(.+))",
        [this](const auto& req, auto& res) { handle_job_files(req, res); });
    
    // Cancel job
    server.Delete(R"(/api/jobs/([a-zA-Z0-9_-]+))",
        [this](const auto& req, auto& res) { handle_cancel_job(req, res); });
    
    // Mesh quality analysis (synchronous)
    server.Post("/api/mesh/quality",
        [this](const auto& req, auto& res) { handle_mesh_quality(req, res); });
    
    // Material library
    server.Get("/api/materials",
        [this](const auto& req, auto& res) { handle_materials(req, res); });
    
    // Health check
    server.Get("/api/health",
        [this](const auto& req, auto& res) { handle_health(req, res); });
    
    // Root endpoint
    server.Get("/", [](const auto& req, auto& res) {
        res.set_content(R"({"service":"FEA Compute Server","version":"1.0.0"})", 
                        "application/json");
    });
    
    // Handle CORS preflight for all routes
    server.Options(".*", [this](const auto& req, auto& res) {
        add_cors_headers(res);
        res.status = 204;
    });
    
    // Error handler
    server.set_error_handler([](const auto& req, auto& res) {
        json error = {
            {"error", "Not found"},
            {"status", res.status},
            {"path", req.path}
        };
        res.set_content(error.dump(), "application/json");
    });
    
    // Start listening (blocking)
    server.listen("0.0.0.0", port);
}

void FEAServer::stop() {
    running = false;
    job_cv.notify_all();
    
    // Wait for workers to finish
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    workers.clear();
    
    server.stop();
}

// ============================================================================
// CORS Headers
// ============================================================================

void FEAServer::add_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
    res.set_header("Access-Control-Max-Age", "86400");
}

// ============================================================================
// Job ID Generation
// ============================================================================

std::string FEAServer::generate_job_id() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    
    std::ostringstream oss;
    oss << "fea_" << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S")
        << "_" << std::setfill('0') << std::setw(6) << dis(gen);
    
    return oss.str();
}

// ============================================================================
// HTTP Handlers
// ============================================================================

void FEAServer::handle_analyze(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    try {
        // Parse JSON body
        json input = json::parse(req.body);
        
        // Validate request
        std::string error;
        if (!validate_request(input, error)) {
            res.status = 400;
            res.set_content(json{
                {"error", error},
                {"status", "validation_error"}
            }.dump(), "application/json");
            return;
        }
        
        // Create job
        AnalysisJob job;
        job.job_id = generate_job_id();
        job.status = JobStatus::QUEUED;
        job.input_data = input;
        job.created_at = std::chrono::system_clock::now();
        job.progress = 0;
        job.current_stage = "Queued";
        job.job_dir = data_dir + "/jobs/" + job.job_id;
        
        // Create job directory
        std::filesystem::create_directories(job.job_dir);
        
        // Save input JSON to job directory
        std::ofstream input_file(job.job_dir + "/input.json");
        input_file << input.dump(2);
        input_file.close();
        
        // Add to queue
        size_t queue_position;
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            jobs[job.job_id] = job;
            job_queue.push(job.job_id);
            queue_position = job_queue.size();
        }
        job_cv.notify_one();
        
        // Return response
        json response = {
            {"job_id", job.job_id},
            {"status", "queued"},
            {"queue_position", queue_position},
            {"message", "Analysis job submitted successfully"},
            {"links", {
                {"status", "/api/jobs/" + job.job_id},
                {"results", "/api/jobs/" + job.job_id + "/results"},
                {"cancel", "/api/jobs/" + job.job_id}
            }}
        };
        
        res.status = 202;  // Accepted
        res.set_content(response.dump(), "application/json");
        
    } catch (const json::parse_error& e) {
        res.status = 400;
        res.set_content(json{
            {"error", "Invalid JSON"},
            {"details", e.what()}
        }.dump(), "application/json");
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{
            {"error", "Internal server error"},
            {"details", e.what()}
        }.dump(), "application/json");
    }
}

void FEAServer::handle_job_status(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    std::string job_id = req.matches[1];
    
    std::lock_guard<std::mutex> lock(jobs_mutex);
    auto it = jobs.find(job_id);
    
    if (it == jobs.end()) {
        res.status = 404;
        res.set_content(json{
            {"error", "Job not found"},
            {"job_id", job_id}
        }.dump(), "application/json");
        return;
    }
    
    const auto& job = it->second;
    res.set_content(job.to_json().dump(), "application/json");
}

void FEAServer::handle_job_results(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    std::string job_id = req.matches[1];
    
    std::lock_guard<std::mutex> lock(jobs_mutex);
    auto it = jobs.find(job_id);
    
    if (it == jobs.end()) {
        res.status = 404;
        res.set_content(json{{"error", "Job not found"}}.dump(), "application/json");
        return;
    }
    
    const auto& job = it->second;
    
    // Check job status
    if (job.status == JobStatus::QUEUED || job.status == JobStatus::RUNNING) {
        res.status = 202;  // Accepted but not complete
        res.set_content(json{
            {"status", to_string(job.status)},
            {"progress", job.progress},
            {"current_stage", job.current_stage},
            {"message", "Job not yet complete"}
        }.dump(), "application/json");
        return;
    }
    
    if (job.status == JobStatus::FAILED) {
        res.status = 400;
        res.set_content(json{
            {"status", "failed"},
            {"error", job.error_message}
        }.dump(), "application/json");
        return;
    }
    
    if (job.status == JobStatus::CANCELLED) {
        res.status = 400;
        res.set_content(json{
            {"status", "cancelled"},
            {"message", "Job was cancelled"}
        }.dump(), "application/json");
        return;
    }
    
    // Return full results
    json response = job.results;
    response["job_id"] = job.job_id;
    response["status"] = "completed";
    response["duration_seconds"] = job.get_duration_seconds();
    
    // Add output file URLs
    json files = json::object();
    if (!job.vtk_path.empty()) {
        files["vtk"] = "/api/jobs/" + job_id + "/files/results.vtu";
    }
    if (!job.csv_path.empty()) {
        files["csv"] = "/api/jobs/" + job_id + "/files/results.csv";
    }
    response["output_files"] = files;
    
    res.set_content(response.dump(), "application/json");
}

void FEAServer::handle_job_files(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    std::string job_id = req.matches[1];
    std::string filename = req.matches[2];
    
    // Security: prevent directory traversal
    if (filename.find("..") != std::string::npos || filename.find('/') == 0) {
        res.status = 400;
        res.set_content(json{{"error", "Invalid filename"}}.dump(), "application/json");
        return;
    }
    
    std::string filepath = data_dir + "/jobs/" + job_id + "/" + filename;
    
    if (!std::filesystem::exists(filepath)) {
        res.status = 404;
        res.set_content(json{{"error", "File not found"}}.dump(), "application/json");
        return;
    }
    
    // Read file
    std::ifstream file(filepath, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    // Determine content type
    std::string ext = std::filesystem::path(filename).extension().string();
    std::string content_type = "application/octet-stream";
    if (ext == ".vtu" || ext == ".vtk") content_type = "application/xml";
    else if (ext == ".csv") content_type = "text/csv";
    else if (ext == ".json") content_type = "application/json";
    else if (ext == ".txt") content_type = "text/plain";
    
    res.set_header("Content-Disposition", "attachment; filename=\"" + filename + "\"");
    res.set_content(content, content_type);
}

void FEAServer::handle_cancel_job(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    std::string job_id = req.matches[1];
    
    std::lock_guard<std::mutex> lock(jobs_mutex);
    auto it = jobs.find(job_id);
    
    if (it == jobs.end()) {
        res.status = 404;
        res.set_content(json{{"error", "Job not found"}}.dump(), "application/json");
        return;
    }
    
    auto& job = it->second;
    
    if (job.status == JobStatus::QUEUED) {
        job.status = JobStatus::CANCELLED;
        job.completed_at = std::chrono::system_clock::now();
        res.set_content(json{
            {"status", "cancelled"},
            {"message", "Job cancelled successfully"}
        }.dump(), "application/json");
    } else if (job.status == JobStatus::RUNNING) {
        // Cannot cancel running job without cooperative cancellation
        res.status = 400;
        res.set_content(json{
            {"error", "Cannot cancel running job"},
            {"message", "Job is currently executing"}
        }.dump(), "application/json");
    } else {
        res.status = 400;
        res.set_content(json{
            {"error", "Job already completed"},
            {"status", to_string(job.status)}
        }.dump(), "application/json");
    }
}

void FEAServer::handle_mesh_quality(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    try {
        json input = json::parse(req.body);
        
        // Create temporary problem to load mesh
        SolverOptions options;
        ElasticProblem<3> problem(options);
        
        parse_mesh(problem, input);
        
        // Get mesh quality
        auto quality = problem.get_mesh_quality();
        
        json response = {
            {"num_elements", quality.num_elements},
            {"num_nodes", quality.num_nodes},
            {"metrics", {
                {"jacobian_ratio", {
                    {"min", quality.min_jacobian_ratio},
                    {"max", 1.0},
                    {"avg", quality.avg_jacobian_ratio}
                }},
                {"aspect_ratio", {
                    {"min", quality.min_aspect_ratio},
                    {"max", quality.max_aspect_ratio},
                    {"avg", quality.avg_aspect_ratio}
                }},
                {"skewness", {
                    {"max", quality.max_skewness},
                    {"avg", quality.avg_skewness}
                }},
                {"warpage", {
                    {"max", quality.max_warpage}
                }}
            }},
            {"poor_quality_elements", quality.num_poor_quality_elements},
            {"poor_quality_percent", 100.0 * quality.num_poor_quality_elements / 
                                     std::max(1u, quality.num_elements)},
            {"quality_acceptable", quality.quality_acceptable},
            {"warnings", quality.warnings},
            {"errors", quality.errors}
        };
        
        res.set_content(response.dump(), "application/json");
        
    } catch (const json::parse_error& e) {
        res.status = 400;
        res.set_content(json{{"error", "Invalid JSON: " + std::string(e.what())}}.dump(),
                        "application/json");
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void FEAServer::handle_health(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    int busy_workers = 0;
    int pending_jobs = 0;
    int completed_jobs = 0;
    int failed_jobs = 0;
    
    {
        std::lock_guard<std::mutex> lock(jobs_mutex);
        pending_jobs = job_queue.size();
        for (const auto& [id, job] : jobs) {
            if (job.status == JobStatus::RUNNING) busy_workers++;
            if (job.status == JobStatus::COMPLETED) completed_jobs++;
            if (job.status == JobStatus::FAILED) failed_jobs++;
        }
    }
    
    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration<double>(now - start_time).count();
    
    json response = {
        {"status", "healthy"},
        {"version", "1.0.0"},
        {"uptime_seconds", uptime},
        {"workers", {
            {"total", num_workers},
            {"busy", busy_workers},
            {"available", num_workers - busy_workers}
        }},
        {"queue", {
            {"pending", pending_jobs},
            {"completed_total", completed_jobs},
            {"failed_total", failed_jobs}
        }},
        {"system", {
            {"data_dir", data_dir}
        }}
    };
    
    res.set_content(response.dump(), "application/json");
}

void FEAServer::handle_materials(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    
    json materials_json = json::array();
    
    for (const auto& mat : material_library.get_all_materials()) {
        json mat_json = {
            {"id", mat.id},
            {"name", mat.name},
            {"model", material_model_to_string(mat.model)}
        };
        
        // Add properties based on material type
        if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat.properties)) {
            mat_json["properties"] = {
                {"youngs_modulus", iso->youngs_modulus},
                {"poissons_ratio", iso->poissons_ratio},
                {"density", iso->density},
                {"thermal_expansion", iso->thermal_expansion_coeff}
            };
        } else if (auto* ortho = std::get_if<OrthotropicElasticProperties>(&mat.properties)) {
            mat_json["properties"] = {
                {"E1", ortho->E1}, {"E2", ortho->E2}, {"E3", ortho->E3},
                {"nu12", ortho->nu12}, {"nu13", ortho->nu13}, {"nu23", ortho->nu23},
                {"G12", ortho->G12}, {"G13", ortho->G13}, {"G23", ortho->G23}
            };
        }
        
        // Add strength data
        if (mat.yield_strength)
            mat_json["yield_strength"] = *mat.yield_strength;
        if (mat.ultimate_strength)
            mat_json["ultimate_strength"] = *mat.ultimate_strength;
        if (mat.fatigue_limit)
            mat_json["fatigue_limit"] = *mat.fatigue_limit;
        
        materials_json.push_back(mat_json);
    }
    
    res.set_content(json{{"materials", materials_json}}.dump(), "application/json");
}

// ============================================================================
// Worker Thread
// ============================================================================

void FEAServer::worker_thread() {
    while (running) {
        std::string job_id;
        
        // Get next job from queue
        {
            std::unique_lock<std::mutex> lock(jobs_mutex);
            job_cv.wait(lock, [this] {
                return !running || !job_queue.empty();
            });
            
            if (!running) break;
            if (job_queue.empty()) continue;
            
            job_id = job_queue.front();
            job_queue.pop();
            
            auto it = jobs.find(job_id);
            if (it == jobs.end()) continue;
            if (it->second.status == JobStatus::CANCELLED) continue;
            
            // Mark as running
            it->second.status = JobStatus::RUNNING;
            it->second.started_at = std::chrono::system_clock::now();
            it->second.current_stage = "Starting analysis";
        }
        
        // Process job outside of lock
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            process_job(jobs[job_id]);
        }
    }
}

void FEAServer::process_job(AnalysisJob& job) {
    try {
        // Parse solver options
        SolverOptions options = parse_solver_options(job.input_data);
        ElasticProblem<3> problem(options);
        
        // Set progress callback
        problem.set_progress_callback([&job](double progress, const std::string& stage) {
            job.progress = progress;
            job.current_stage = stage;
        });
        
        // Setup unit system
        if (job.input_data.contains("units")) {
            problem.set_unit_system(parse_unit_system(job.input_data["units"]));
        }
        
        // Set material library
        problem.set_material_library(material_library);
        
        // Parse and setup mesh
        job.current_stage = "Loading mesh";
        parse_mesh(problem, job.input_data);
        
        // Parse and setup materials
        job.current_stage = "Setting up materials";
        parse_materials(problem, job.input_data);
        
        // Parse and setup boundary conditions
        job.current_stage = "Applying boundary conditions";
        parse_boundary_conditions(problem, job.input_data);
        
        // Parse and setup loads
        job.current_stage = "Applying loads";
        parse_loads(problem, job.input_data);
        
        // Parse connections if present
        if (job.input_data.contains("connections")) {
            job.current_stage = "Setting up connections";
            parse_connections(problem, job.input_data);
        }
        
        // Parse linearized stress lines if present
        if (job.input_data.contains("linearized_stress_lines")) {
            parse_linearized_stress_lines(problem, job.input_data);
        }
        
        // Run analysis
        job.current_stage = "Running analysis";
        problem.run();
        
        // Get results
        job.current_stage = "Collecting results";
        const auto& results = problem.get_results();
        job.results = results.to_json();
        
        // Save output files
        job.current_stage = "Writing output files";
        std::string vtk_file = job.job_dir + "/results.vtu";
        problem.output_vtk(vtk_file);
        job.vtk_path = vtk_file;
        
        // Save results JSON
        std::ofstream results_file(job.job_dir + "/results.json");
        results_file << job.results.dump(2);
        results_file.close();
        
        // Mark complete
        job.status = JobStatus::COMPLETED;
        job.completed_at = std::chrono::system_clock::now();
        job.progress = 1.0;
        job.current_stage = "Completed";
        
    } catch (const std::exception& e) {
        job.status = JobStatus::FAILED;
        job.error_message = e.what();
        job.completed_at = std::chrono::system_clock::now();
        job.current_stage = "Failed: " + std::string(e.what());
        
        // Log error to file
        std::ofstream error_file(job.job_dir + "/error.log");
        error_file << "Error: " << e.what() << "\n";
        error_file << "Stage: " << job.current_stage << "\n";
        error_file.close();
    }
}

// ============================================================================
// Request Validation
// ============================================================================

bool FEAServer::validate_request(const json& data, std::string& error) {
    // Check required fields
    if (!data.contains("mesh")) {
        error = "Missing required field: mesh";
        return false;
    }
    
    if (!data.contains("boundary_conditions")) {
        error = "Missing required field: boundary_conditions";
        return false;
    }
    
    // Validate mesh
    if (!validate_mesh(data["mesh"], error)) {
        return false;
    }
    
    // Validate boundary conditions
    if (!validate_boundary_conditions(data["boundary_conditions"], error)) {
        return false;
    }
    
    // Validate loads if present
    if (data.contains("loads")) {
        if (!validate_loads(data["loads"], error)) {
            return false;
        }
    }
    
    return true;
}

bool FEAServer::validate_mesh(const json& mesh, std::string& error) {
    if (!mesh.contains("type")) {
        error = "mesh.type is required";
        return false;
    }
    
    std::string mesh_type = mesh["type"];
    
    if (mesh_type == "box") {
        if (!mesh.contains("min") || !mesh.contains("max")) {
            error = "box mesh requires 'min' and 'max' points";
            return false;
        }
        if (!mesh["min"].is_array() || mesh["min"].size() != 3) {
            error = "mesh.min must be array of 3 coordinates";
            return false;
        }
        if (!mesh["max"].is_array() || mesh["max"].size() != 3) {
            error = "mesh.max must be array of 3 coordinates";
            return false;
        }
    } else if (mesh_type == "cylinder") {
        if (!mesh.contains("radius") || !mesh.contains("height")) {
            error = "cylinder mesh requires 'radius' and 'height'";
            return false;
        }
    } else if (mesh_type == "file") {
        if (!mesh.contains("data") && !mesh.contains("url") && !mesh.contains("path")) {
            error = "file mesh requires 'data', 'url', or 'path'";
            return false;
        }
    } else {
        error = "Unknown mesh type: " + mesh_type;
        return false;
    }
    
    return true;
}

bool FEAServer::validate_boundary_conditions(const json& bcs, std::string& error) {
    if (!bcs.is_array()) {
        error = "boundary_conditions must be an array";
        return false;
    }
    
    if (bcs.empty()) {
        error = "boundary_conditions must not be empty";
        return false;
    }
    
    for (size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        if (!bc.contains("type")) {
            error = "boundary_conditions[" + std::to_string(i) + "] missing 'type'";
            return false;
        }
        
        std::string type = bc["type"];
        if (type != "fixed" && type != "displacement" && type != "symmetry" &&
            type != "elastic_support" && type != "frictionless") {
            error = "Unknown boundary condition type: " + type;
            return false;
        }
    }
    
    return true;
}

bool FEAServer::validate_loads(const json& loads, std::string& error) {
    if (!loads.is_array()) {
        error = "loads must be an array";
        return false;
    }
    
    for (size_t i = 0; i < loads.size(); ++i) {
        const auto& load = loads[i];
        if (!load.contains("type")) {
            error = "loads[" + std::to_string(i) + "] missing 'type'";
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// JSON Parsing Helpers
// ============================================================================

SolverOptions FEAServer::parse_solver_options(const json& data) {
    SolverOptions opts;
    
    if (!data.contains("solver_options")) return opts;
    
    const auto& so = data["solver_options"];
    
    if (so.contains("fe_degree"))
        opts.fe_degree = so["fe_degree"];
    if (so.contains("refinement_cycles"))
        opts.refinement_cycles = so["refinement_cycles"];
    if (so.contains("adaptive_refinement"))
        opts.adaptive_refinement = so["adaptive_refinement"];
    if (so.contains("max_adaptive_cycles"))
        opts.max_adaptive_cycles = so["max_adaptive_cycles"];
    if (so.contains("max_iterations"))
        opts.max_iterations = so["max_iterations"];
    if (so.contains("tolerance"))
        opts.tolerance = so["tolerance"];
    if (so.contains("large_deformation"))
        opts.large_deformation = so["large_deformation"];
    if (so.contains("compute_reactions"))
        opts.compute_reactions = so["compute_reactions"];
    if (so.contains("compute_safety_factors"))
        opts.compute_safety_factors = so["compute_safety_factors"];
    if (so.contains("output_vtk"))
        opts.output_vtk = so["output_vtk"];
    
    return opts;
}

UnitSystem FEAServer::parse_unit_system(const json& data) {
    std::string type = data.value("type", "SI");
    
    if (type == "SI") return UnitSystem::SI();
    if (type == "SI_MM") return UnitSystem::SI_MM();
    if (type == "US_CUSTOMARY") return UnitSystem::US_Customary();
    
    // Custom unit system
    if (type == "custom" && data.contains("factors")) {
        UnitSystem us;
        const auto& f = data["factors"];
        us.length_to_si = f.value("length", 1.0);
        us.force_to_si = f.value("force", 1.0);
        us.pressure_to_si = f.value("pressure", 1.0);
        us.temperature_offset = f.value("temperature_offset", 0.0);
        return us;
    }
    
    return UnitSystem::SI();
}

void FEAServer::parse_mesh(ElasticProblem<3>& problem, const json& data) {
    const auto& mesh = data["mesh"];
    std::string type = mesh["type"];
    
    if (type == "box") {
        auto min_pt = mesh["min"].get<std::vector<double>>();
        auto max_pt = mesh["max"].get<std::vector<double>>();
        
        std::vector<unsigned int> subdivisions = {4, 4, 4};
        if (mesh.contains("subdivisions")) {
            subdivisions = mesh["subdivisions"].get<std::vector<unsigned int>>();
        }
        
        problem.generate_box_mesh(
            Point<3>(min_pt[0], min_pt[1], min_pt[2]),
            Point<3>(max_pt[0], max_pt[1], max_pt[2]),
            subdivisions);
            
    } else if (type == "cylinder") {
        Point<3> center(0, 0, 0);
        if (mesh.contains("center")) {
            auto c = mesh["center"].get<std::vector<double>>();
            center = Point<3>(c[0], c[1], c[2]);
        }
        
        double radius = mesh["radius"];
        double height = mesh["height"];
        
        unsigned int n_radial = mesh.value("n_radial", 8);
        unsigned int n_axial = mesh.value("n_axial", 4);
        
        problem.generate_cylinder_mesh(center, radius, height, n_radial, n_axial);
        
    } else if (type == "file") {
        std::string format = mesh.value("format", "msh");
        
        if (mesh.contains("data")) {
            // Mesh data embedded in JSON
            std::string mesh_data = mesh["data"];
            problem.read_mesh_from_string(mesh_data, format);
        } else if (mesh.contains("path")) {
            // Read from local file
            problem.read_mesh(mesh["path"]);
        }
    }
    
    // Apply mesh transformations
    if (mesh.contains("scale")) {
        double scale = mesh["scale"];
        problem.scale_mesh(scale);
    }
}

void FEAServer::parse_materials(ElasticProblem<3>& problem, const json& data) {
    if (!data.contains("materials")) {
        // Use default steel
        problem.set_default_material(MaterialLibrary::STEEL_STRUCTURAL);
        return;
    }
    
    const auto& mats = data["materials"];
    
    // Set default material
    if (mats.contains("default")) {
        problem.set_default_material(mats["default"]);
    }
    
    // Assign materials to regions
    if (mats.contains("regions")) {
        for (const auto& region : mats["regions"]) {
            unsigned int region_id = region["region_id"];
            std::string material = region["material"];
            problem.assign_material_to_region(region_id, material);
        }
    }
    
    // Add custom materials
    if (mats.contains("custom")) {
        for (const auto& custom : mats["custom"]) {
            Material mat;
            mat.name = custom["name"];
            mat.id = custom.value("id", 0);
            
            std::string model = custom.value("model", "isotropic_elastic");
            if (model == "isotropic_elastic") {
                mat.model = MaterialModel::IsotropicElastic;
                IsotropicElasticProperties props;
                props.youngs_modulus = custom["youngs_modulus"];
                props.poissons_ratio = custom["poissons_ratio"];
                props.density = custom.value("density", 0.0);
                props.thermal_expansion_coeff = custom.value("thermal_expansion", 0.0);
                mat.properties = props;
            }
            
            if (custom.contains("yield_strength"))
                mat.yield_strength = custom["yield_strength"];
            if (custom.contains("ultimate_strength"))
                mat.ultimate_strength = custom["ultimate_strength"];
            
            problem.add_custom_material(mat);
        }
    }
}

void FEAServer::parse_boundary_conditions(ElasticProblem<3>& problem, const json& data) {
    for (const auto& bc_json : data["boundary_conditions"]) {
        std::string type = bc_json["type"];
        
        // Parse target
        BoundaryTarget target;
        if (bc_json.contains("target")) {
            const auto& t = bc_json["target"];
            std::string target_type = t["type"];
            
            if (target_type == "boundary_id") {
                target = BoundaryTarget::from_boundary_id(t["id"]);
            } else if (target_type == "point") {
                auto loc = t["location"].get<std::vector<double>>();
                double tol = t.value("tolerance", 1e-6);
                target = BoundaryTarget::from_point(Point<3>(loc[0], loc[1], loc[2]), tol);
            } else if (target_type == "box") {
                auto min_pt = t["min"].get<std::vector<double>>();
                auto max_pt = t["max"].get<std::vector<double>>();
                target = BoundaryTarget::from_box(
                    Point<3>(min_pt[0], min_pt[1], min_pt[2]),
                    Point<3>(max_pt[0], max_pt[1], max_pt[2]));
            } else if (target_type == "plane") {
                auto pt = t["point"].get<std::vector<double>>();
                auto norm = t["normal"].get<std::vector<double>>();
                double tol = t.value("tolerance", 1e-6);
                target = BoundaryTarget::from_plane(
                    Point<3>(pt[0], pt[1], pt[2]),
                    Tensor<1, 3>({norm[0], norm[1], norm[2]}),
                    tol);
            }
        }
        
        std::string desc = bc_json.value("description", "");
        
        // Create appropriate BC type
        if (type == "fixed") {
            problem.add_boundary_condition(FixedBC(target, desc));
        }
        else if (type == "displacement") {
            DisplacementBC bc;
            bc.target = target;
            bc.description = desc;
            
            auto values = bc_json["values"];
            for (int i = 0; i < 3; ++i) {
                if (!values[i].is_null()) {
                    bc.values[i] = values[i].get<double>();
                } else {
                    bc.values[i] = std::nullopt;
                }
            }
            
            problem.add_boundary_condition(bc);
        }
        else if (type == "symmetry") {
            auto normal = bc_json["plane_normal"].get<std::vector<double>>();
            problem.add_boundary_condition(SymmetryBC(
                target,
                Tensor<1, 3>({normal[0], normal[1], normal[2]}),
                desc));
        }
        else if (type == "elastic_support") {
            ElasticSupportBC bc;
            bc.target = target;
            bc.description = desc;
            
            if (bc_json.contains("stiffness_per_area")) {
                auto k = bc_json["stiffness_per_area"].get<std::vector<double>>();
                for (int i = 0; i < 3; ++i) bc.stiffness_per_area[i] = k[i];
            }
            
            problem.add_boundary_condition(bc);
        }
    }
}

void FEAServer::parse_loads(ElasticProblem<3>& problem, const json& data) {
    if (!data.contains("loads")) return;
    
    for (const auto& load_json : data["loads"]) {
        std::string type = load_json["type"];
        std::string desc = load_json.value("description", "");
        
        if (type == "gravity") {
            auto accel = load_json["acceleration"].get<std::vector<double>>();
            problem.add_load(GravityLoad(
                Tensor<1, 3>({accel[0], accel[1], accel[2]}), desc));
        }
        else if (type == "surface_force") {
            BoundaryTarget target = BoundaryTarget::from_boundary_id(
                load_json["target"]["id"]);
            auto force = load_json["force_per_area"].get<std::vector<double>>();
            problem.add_load(SurfaceForceLoad(
                target, Tensor<1, 3>({force[0], force[1], force[2]}), desc));
        }
        else if (type == "pressure") {
            BoundaryTarget target = BoundaryTarget::from_boundary_id(
                load_json["target"]["id"]);
            double pressure = load_json["value"];
            
            PressureLoad pl(target, pressure, desc);
            pl.is_follower = load_json.value("is_follower", false);
            problem.add_load(pl);
        }
        else if (type == "point_force") {
            auto loc = load_json["location"].get<std::vector<double>>();
            auto force = load_json["force"].get<std::vector<double>>();
            
            PointForceLoad pf;
            pf.location = Point<3>(loc[0], loc[1], loc[2]);
            pf.force = Tensor<1, 3>({force[0], force[1], force[2]});
            pf.distribution_radius = load_json.value("distribution_radius", 0.0);
            pf.description = desc;
            problem.add_load(pf);
        }
        else if (type == "moment") {
            auto loc = load_json["location"].get<std::vector<double>>();
            auto moment = load_json["moment"].get<std::vector<double>>();
            
            PointMomentLoad pm;
            pm.location = Point<3>(loc[0], loc[1], loc[2]);
            pm.moment = Tensor<1, 3>({moment[0], moment[1], moment[2]});
            pm.distribution_radius = load_json.value("distribution_radius", 0.01);
            pm.description = desc;
            problem.add_load(pm);
        }
        else if (type == "thermal") {
            double t_ref = load_json["reference_temperature"];
            double t_applied = load_json["applied_temperature"];
            problem.add_load(UniformThermalLoad(t_ref, t_applied, desc));
        }
        else if (type == "centrifugal") {
            auto axis = load_json["axis"].get<std::vector<double>>();
            auto center = load_json.value("center", std::vector<double>{0,0,0});
            double omega = load_json["angular_velocity"];
            
            CentrifugalLoad cl;
            cl.axis = Tensor<1, 3>({axis[0], axis[1], axis[2]});
            cl.center = Point<3>(center[0], center[1], center[2]);
            cl.angular_velocity = omega;
            cl.description = desc;
            problem.add_load(cl);
        }
    }
}

void FEAServer::parse_connections(ElasticProblem<3>& problem, const json& data) {
    if (!data.contains("connections")) return;
    
    for (const auto& conn_json : data["connections"]) {
        std::string type = conn_json["type"];
        std::string desc = conn_json.value("description", "");
        
        if (type == "spring_to_ground") {
            auto loc = conn_json["location"].get<std::vector<double>>();
            auto stiff = conn_json["stiffness"].get<std::vector<double>>();
            
            SpringToGroundConnection spring;
            spring.location = Point<3>(loc[0], loc[1], loc[2]);
            for (int i = 0; i < 3; ++i) spring.stiffness[i] = stiff[i];
            spring.tolerance = conn_json.value("tolerance", 1e-6);
            spring.description = desc;
            problem.add_connection(spring);
        }
        else if (type == "rigid") {
            auto master = conn_json["master"]["location"].get<std::vector<double>>();
            
            RigidConnection rigid;
            rigid.master_point = Point<3>(master[0], master[1], master[2]);
            rigid.master_tolerance = conn_json["master"].value("tolerance", 1e-6);
            rigid.slave_target = BoundaryTarget::from_boundary_id(
                conn_json["slave"]["boundary_id"]);
            rigid.type = RigidConnection::Type::RBE2;
            rigid.description = desc;
            problem.add_connection(rigid);
        }
    }
}

void FEAServer::parse_linearized_stress_lines(ElasticProblem<3>& problem, const json& data) {
    if (!data.contains("linearized_stress_lines")) return;
    
    for (const auto& scl_json : data["linearized_stress_lines"]) {
        auto start = scl_json["start"].get<std::vector<double>>();
        auto end = scl_json["end"].get<std::vector<double>>();
        std::string name = scl_json.value("name", "");
        unsigned int n_points = scl_json.value("num_points", 20);
        
        problem.add_stress_classification_line(
            Point<3>(start[0], start[1], start[2]),
            Point<3>(end[0], end[1], end[2]),
            name, n_points);
    }
}

// ============================================================================
// Job Cleanup
// ============================================================================

void FEAServer::cleanup_old_jobs(int max_age_hours) {
    auto now = std::chrono::system_clock::now();
    auto max_age = std::chrono::hours(max_age_hours);
    
    std::lock_guard<std::mutex> lock(jobs_mutex);
    
    std::vector<std::string> to_remove;
    for (const auto& [id, job] : jobs) {
        if (job.status == JobStatus::COMPLETED || job.status == JobStatus::FAILED ||
            job.status == JobStatus::CANCELLED) {
            auto age = now - job.completed_at;
            if (age > max_age) {
                to_remove.push_back(id);
            }
        }
    }
    
    for (const auto& id : to_remove) {
        // Remove job directory
        std::filesystem::remove_all(jobs[id].job_dir);
        jobs.erase(id);
    }
}

} // namespace FEA
