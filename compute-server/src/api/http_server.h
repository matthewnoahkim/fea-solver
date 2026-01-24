#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

/**
 * @file http_server.h
 * @brief HTTP REST API server for FEA analysis
 * 
 * Provides a RESTful API interface for submitting and managing FEA jobs:
 * - POST /api/analyze - Submit new analysis job
 * - GET /api/jobs/{id} - Get job status
 * - GET /api/jobs/{id}/results - Get job results
 * - GET /api/jobs/{id}/files/{filename} - Download output files
 * - DELETE /api/jobs/{id} - Cancel job
 * - POST /api/mesh/quality - Analyze mesh quality
 * - GET /api/materials - List available materials
 * - GET /api/health - Server health check
 */

#include "../solver/elastic_problem.h"
#include "../solver/material_library.h"
#include "../solver/boundary_conditions.h"
#include "../solver/loads/load_base.h"
#include "../solver/connections/constraint_base.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>

namespace FEA {

using json = nlohmann::json;

// ============================================================================
// Job Status
// ============================================================================

/**
 * @brief Analysis job status
 */
enum class JobStatus {
    QUEUED,     ///< Waiting in queue
    RUNNING,    ///< Currently executing
    COMPLETED,  ///< Successfully finished
    FAILED,     ///< Failed with error
    CANCELLED   ///< Cancelled by user
};

std::string to_string(JobStatus status);
JobStatus status_from_string(const std::string& str);

// ============================================================================
// Analysis Job
// ============================================================================

/**
 * @brief Represents an analysis job with its state and results
 */
struct AnalysisJob {
    std::string job_id;                         ///< Unique job identifier
    JobStatus status;                           ///< Current status
    json input_data;                            ///< Original request JSON
    json results;                               ///< Analysis results
    std::string error_message;                  ///< Error details if failed
    
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point completed_at;
    
    double progress;                            ///< Progress 0.0 to 1.0
    std::string current_stage;                  ///< Current operation description
    
    std::string vtk_path;                       ///< Path to VTK output file
    std::string csv_path;                       ///< Path to CSV output file
    std::string job_dir;                        ///< Job working directory
    
    /**
     * @brief Get job duration in seconds
     */
    double get_duration_seconds() const {
        auto end = (status == JobStatus::COMPLETED || status == JobStatus::FAILED) 
                   ? completed_at : std::chrono::system_clock::now();
        auto start = (status == JobStatus::QUEUED) ? created_at : started_at;
        return std::chrono::duration<double>(end - start).count();
    }
    
    /**
     * @brief Convert job to JSON for API response
     */
    json to_json() const {
        json j;
        j["job_id"] = job_id;
        j["status"] = to_string(status);
        j["progress"] = progress;
        j["current_stage"] = current_stage;
        j["created_at"] = std::chrono::duration_cast<std::chrono::seconds>(
            created_at.time_since_epoch()).count();
        j["duration_seconds"] = get_duration_seconds();
        
        if (status == JobStatus::COMPLETED) {
            j["completed_at"] = std::chrono::duration_cast<std::chrono::seconds>(
                completed_at.time_since_epoch()).count();
        }
        
        if (status == JobStatus::FAILED) {
            j["error"] = error_message;
        }
        
        return j;
    }
};

// ============================================================================
// FEA Server
// ============================================================================

/**
 * @brief HTTP server for FEA analysis API
 * 
 * Manages a pool of worker threads that process analysis jobs asynchronously.
 * Jobs are submitted via REST API and results retrieved when complete.
 * 
 * Example usage:
 * @code
 * FEAServer server(8080, 4, "/data");
 * server.start();  // Blocks until stopped
 * @endcode
 */
class FEAServer {
public:
    /**
     * @brief Construct server
     * @param port HTTP port to listen on
     * @param num_workers Number of worker threads
     * @param data_dir Directory for job data storage
     */
    FEAServer(int port = 8080, 
              int num_workers = 4,
              const std::string& data_dir = "/data");
    
    ~FEAServer();
    
    /**
     * @brief Start server (blocking)
     */
    void start();
    
    /**
     * @brief Stop server and workers
     */
    void stop();
    
    /**
     * @brief Check if server is running
     */
    bool is_running() const { return running; }
    
    /**
     * @brief Get server port
     */
    int get_port() const { return port; }
    
    /**
     * @brief Set maximum job age before cleanup (hours)
     */
    void set_max_job_age(int hours) { max_job_age_hours = hours; }
    
    /**
     * @brief Get material library reference
     */
    MaterialLibrary& get_material_library() { return material_library; }
    
private:
    // =========================================================================
    // HTTP Request Handlers
    // =========================================================================
    
    void handle_analyze(const httplib::Request& req, httplib::Response& res);
    void handle_job_status(const httplib::Request& req, httplib::Response& res);
    void handle_job_results(const httplib::Request& req, httplib::Response& res);
    void handle_job_files(const httplib::Request& req, httplib::Response& res);
    void handle_cancel_job(const httplib::Request& req, httplib::Response& res);
    void handle_mesh_quality(const httplib::Request& req, httplib::Response& res);
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_materials(const httplib::Request& req, httplib::Response& res);
    
    // =========================================================================
    // Job Management
    // =========================================================================
    
    /**
     * @brief Worker thread function
     */
    void worker_thread();
    
    /**
     * @brief Process a single analysis job
     */
    void process_job(AnalysisJob& job);
    
    /**
     * @brief Generate unique job ID
     */
    std::string generate_job_id();
    
    /**
     * @brief Remove old completed jobs
     */
    void cleanup_old_jobs(int max_age_hours = 24);
    
    // =========================================================================
    // JSON Parsing
    // =========================================================================
    
    SolverOptions parse_solver_options(const json& data);
    UnitSystem parse_unit_system(const json& data);
    void parse_mesh(ElasticProblem<3>& problem, const json& data);
    void parse_materials(ElasticProblem<3>& problem, const json& data);
    void parse_boundary_conditions(ElasticProblem<3>& problem, const json& data);
    void parse_loads(ElasticProblem<3>& problem, const json& data);
    void parse_connections(ElasticProblem<3>& problem, const json& data);
    void parse_linearized_stress_lines(ElasticProblem<3>& problem, const json& data);
    
    // =========================================================================
    // Validation
    // =========================================================================
    
    bool validate_request(const json& data, std::string& error);
    bool validate_mesh(const json& mesh, std::string& error);
    bool validate_boundary_conditions(const json& bcs, std::string& error);
    bool validate_loads(const json& loads, std::string& error);
    
    // =========================================================================
    // CORS
    // =========================================================================
    
    void add_cors_headers(httplib::Response& res);
    
    // =========================================================================
    // Member Variables
    // =========================================================================
    
    httplib::Server server;
    int port;
    std::string data_dir;
    std::atomic<bool> running;
    
    // Job storage
    std::unordered_map<std::string, AnalysisJob> jobs;
    std::queue<std::string> job_queue;
    std::mutex jobs_mutex;
    std::condition_variable job_cv;
    
    // Worker threads
    std::vector<std::thread> workers;
    int num_workers;
    
    // Material library
    MaterialLibrary material_library;
    
    // Configuration
    int max_job_age_hours = 24;
    
    // Server start time for uptime tracking
    std::chrono::system_clock::time_point start_time;
};

} // namespace FEA

#endif // HTTP_SERVER_H
