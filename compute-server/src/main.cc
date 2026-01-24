/**
 * @file main.cc
 * @brief Entry point for FEA Compute Server
 * 
 * Starts the HTTP server that provides REST API access to the
 * finite element analysis solver. Handles command-line arguments
 * and signal-based shutdown.
 * 
 * Usage:
 *   fea_server [options]
 *   
 * Options:
 *   --port <port>       Server port (default: 8080)
 *   --workers <n>       Number of worker threads (default: 4)
 *   --data-dir <path>   Data directory (default: /data)
 *   --help              Show help message
 */

#include "api/http_server.h"
#include <iostream>
#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>

// Global server pointer for signal handler
FEA::FEAServer* global_server = nullptr;

/**
 * @brief Signal handler for graceful shutdown
 */
void signal_handler(int signum) {
    std::cout << "\n[Server] Received signal " << signum << ", initiating shutdown..." << std::endl;
    
    if (global_server) {
        global_server->stop();
    }
    
    std::cout << "[Server] Shutdown complete." << std::endl;
    exit(signum);
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "FEA Compute Server v1.0.0\n"
              << "A production-grade 3D static structural finite element analysis server\n\n"
              << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  --port <port>       HTTP server port (default: 8080)\n"
              << "  --workers <n>       Number of worker threads (default: 4)\n"
              << "  --data-dir <path>   Directory for job data storage (default: /data)\n"
              << "  --help, -h          Show this help message\n\n"
              << "Environment Variables:\n"
              << "  FEA_PORT            Alternative to --port\n"
              << "  FEA_WORKERS         Alternative to --workers\n"
              << "  FEA_DATA_DIR        Alternative to --data-dir\n\n"
              << "API Endpoints:\n"
              << "  POST /api/analyze              Submit analysis job\n"
              << "  GET  /api/jobs/{id}            Get job status\n"
              << "  GET  /api/jobs/{id}/results    Get analysis results\n"
              << "  GET  /api/jobs/{id}/files/{f}  Download output file\n"
              << "  DELETE /api/jobs/{id}          Cancel job\n"
              << "  POST /api/mesh/quality         Analyze mesh quality\n"
              << "  GET  /api/materials            List available materials\n"
              << "  GET  /api/health               Server health check\n\n"
              << "Example:\n"
              << "  " << program_name << " --port 8080 --workers 8 --data-dir ./data\n";
}

/**
 * @brief Print server startup banner
 */
void print_banner(int port, int workers, const std::string& data_dir) {
    std::cout << R"(
 _____ _____    _      ____                      _            
|  ___|  ___|  / \    / ___|  ___ _ ____   _____| |           
| |_  | |_    / _ \   \___ \ / _ \ '__\ \ / / _ \ |           
|  _| |  _|  / ___ \   ___) |  __/ |   \ V /  __/ |           
|_|   |_|   /_/   \_\ |____/ \___|_|    \_/ \___|_|           
                                                              
        Compute Server v1.0.0 - 3D Static Structural FEA
)" << std::endl;
    
    std::cout << "Configuration:\n"
              << "  Port:       " << port << "\n"
              << "  Workers:    " << workers << "\n"
              << "  Data Dir:   " << data_dir << "\n"
              << "  Threads:    " << std::thread::hardware_concurrency() << " available\n"
              << std::endl;
}

/**
 * @brief Get environment variable with default value
 */
std::string get_env(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : default_value;
}

int get_env_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value) {
        try {
            return std::stoi(value);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

int main(int argc, char* argv[]) {
    // Default configuration from environment or hardcoded defaults
    int port = get_env_int("FEA_PORT", 8080);
    int workers = get_env_int("FEA_WORKERS", 4);
    std::string data_dir = get_env("FEA_DATA_DIR", "/data");
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--port" && i + 1 < argc) {
            try {
                port = std::stoi(argv[++i]);
                if (port < 1 || port > 65535) {
                    std::cerr << "Error: Port must be between 1 and 65535\n";
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid port number\n";
                return 1;
            }
        }
        else if (arg == "--workers" && i + 1 < argc) {
            try {
                workers = std::stoi(argv[++i]);
                if (workers < 1) {
                    std::cerr << "Error: Workers must be at least 1\n";
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid worker count\n";
                return 1;
            }
        }
        else if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
    }
    
    // Print banner
    print_banner(port, workers, data_dir);
    
    // Setup signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // Docker stop
#ifdef SIGQUIT
    signal(SIGQUIT, signal_handler);  // Quit signal
#endif
    
    try {
        // Create and start server
        std::cout << "[Server] Initializing..." << std::endl;
        
        FEA::FEAServer server(port, workers, data_dir);
        global_server = &server;
        
        std::cout << "[Server] Ready" << std::endl;
        std::cout << "[Server] Listening on http://0.0.0.0:" << port << std::endl;
        std::cout << "[Server] Press Ctrl+C to shutdown\n" << std::endl;
        
        // Start server (blocking)
        server.start();
        
    } catch (const std::exception& e) {
        std::cerr << "[Server] Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
