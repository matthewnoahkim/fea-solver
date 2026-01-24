# =============================================================================
# FEA Compute Server - CMake Build Configuration
# =============================================================================
# 
# A production-grade 3D static structural FEA backend using deal.ii
#
# Build Instructions:
#   mkdir build && cd build
#   cmake .. -DDEAL_II_DIR=/path/to/dealii
#   make -j$(nproc)
#
# =============================================================================

cmake_minimum_required(VERSION 3.13)
project(fea_server VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# C++ Standard Settings
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Find deal.II
# =============================================================================

find_package(deal.II 9.4 REQUIRED
    HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)
deal_ii_initialize_cached_variables()

# Check for required deal.II features
if(NOT DEAL_II_WITH_LAPACK)
    message(FATAL_ERROR "
    ==========================================
    ERROR: deal.II must be compiled with LAPACK support.
    Please reconfigure deal.II with -DDEAL_II_WITH_LAPACK=ON
    ==========================================
    ")
endif()

# Optional features with warnings
if(NOT DEAL_II_WITH_UMFPACK)
    message(WARNING "deal.II compiled without UMFPACK - using iterative solvers only")
endif()

if(NOT DEAL_II_WITH_P4EST)
    message(WARNING "deal.II compiled without p4est - distributed meshes not available")
endif()

# =============================================================================
# Fetch External Dependencies
# =============================================================================

include(FetchContent)

# cpp-httplib - Header-only HTTP server
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.14.3
)
FetchContent_MakeAvailable(httplib)

# nlohmann/json - JSON library
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# =============================================================================
# Source Files
# =============================================================================

# Solver core
set(SOLVER_SOURCES
    src/solver/elastic_problem.cc
    src/solver/material_library.cc
    src/solver/boundary_conditions.cc
    src/solver/loads/load_manager.cc
    src/solver/connections/connection_manager.cc
    src/solver/nonlinear/newton_raphson.cc
    src/solver/nonlinear/line_search.cc
    src/solver/material_models/linear_elastic.cc
    src/solver/material_models/orthotropic.cc
    src/solver/material_models/elastoplastic.cc
    src/solver/material_models/hyperelastic.cc
)

# Mesh handling
set(MESH_SOURCES
    src/mesh/mesh_generator.cc
    src/mesh/mesh_reader.cc
    src/mesh/mesh_quality.cc
    src/mesh/region_manager.cc
)

# Elements
set(ELEMENT_SOURCES
    src/elements/solid_element.cc
    src/elements/shell_element.cc
    src/elements/beam_element.cc
)

# Post-processing
set(POST_SOURCES
    src/post/stress_calculator.cc
    src/post/reaction_forces.cc
    src/post/safety_factors.cc
    src/post/linearized_stress.cc
    src/post/section_cuts.cc
    src/post/strain_energy.cc
    src/post/vtk_output.cc
)

# Utilities
set(UTIL_SOURCES
    src/utils/tensor_utils.cc
    src/utils/unit_converter.cc
)

# API server
set(API_SOURCES
    src/api/http_server.cc
    src/api/json_parser.cc
    src/api/request_validator.cc
)

# All sources
set(ALL_SOURCES
    src/main.cc
    ${SOLVER_SOURCES}
    ${MESH_SOURCES}
    ${ELEMENT_SOURCES}
    ${POST_SOURCES}
    ${UTIL_SOURCES}
    ${API_SOURCES}
)

# =============================================================================
# Main Executable
# =============================================================================

add_executable(fea_server ${ALL_SOURCES})

# Setup deal.II for target
deal_ii_setup_target(fea_server)

# Link libraries
target_link_libraries(fea_server PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
)

# Include directories
target_include_directories(fea_server PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# =============================================================================
# Threading Support
# =============================================================================

find_package(Threads REQUIRED)
target_link_libraries(fea_server PRIVATE Threads::Threads)

# =============================================================================
# Filesystem Support (for older compilers)
# =============================================================================

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(fea_server PRIVATE stdc++fs)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(fea_server PRIVATE c++fs)
    endif()
endif()

# =============================================================================
# Compiler Flags
# =============================================================================

# Release optimization
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(fea_server PRIVATE 
        -O3 
        -DNDEBUG
        -ffast-math
    )
    
    # Enable native CPU optimization if not cross-compiling
    if(NOT CMAKE_CROSSCOMPILING)
        target_compile_options(fea_server PRIVATE -march=native)
    endif()
endif()

# Debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(fea_server PRIVATE 
        -g 
        -O0
        -Wall 
        -Wextra 
        -Wpedantic
        -Wconversion
        -Wshadow
    )
    
    # Enable address sanitizer in debug mode
    option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
    if(ENABLE_ASAN)
        target_compile_options(fea_server PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(fea_server PRIVATE -fsanitize=address)
    endif()
endif()

# RelWithDebInfo
if(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    target_compile_options(fea_server PRIVATE 
        -O2 
        -g
        -DNDEBUG
    )
endif()

# =============================================================================
# OpenMP Support (Optional)
# =============================================================================

option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fea_server PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(fea_server PRIVATE FEA_WITH_OPENMP)
    else()
        message(WARNING "OpenMP not found - building without OpenMP support")
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS fea_server 
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/data/
    DESTINATION share/fea_server/data
    OPTIONAL
)

# =============================================================================
# Tests (Optional)
# =============================================================================

option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    
    # Try to find GTest
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    
    add_subdirectory(tests)
endif()

# =============================================================================
# Benchmark (Optional)
# =============================================================================

option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(BUILD_BENCHMARKS)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
    
    # Add benchmark executable
    add_executable(fea_benchmark tests/benchmarks/benchmark_solver.cc)
    deal_ii_setup_target(fea_benchmark)
    target_link_libraries(fea_benchmark PRIVATE benchmark::benchmark)
endif()

# =============================================================================
# Configuration Summary
# =============================================================================

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════════╗")
message(STATUS "║          FEA Server Configuration Summary                    ║")
message(STATUS "╠══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Version:        ${PROJECT_VERSION}                                       ║")
message(STATUS "║  Build type:     ${CMAKE_BUILD_TYPE}                                    ║")
message(STATUS "║  C++ Standard:   C++${CMAKE_CXX_STANDARD}                                        ║")
message(STATUS "╠══════════════════════════════════════════════════════════════╣")
message(STATUS "║  deal.II:                                                    ║")
message(STATUS "║    Version:      ${DEAL_II_VERSION}                                       ║")
message(STATUS "║    With MPI:     ${DEAL_II_WITH_MPI}                                         ║")
message(STATUS "║    With PETSc:   ${DEAL_II_WITH_PETSC}                                         ║")
message(STATUS "║    With Trilinos:${DEAL_II_WITH_TRILINOS}                                         ║")
message(STATUS "║    With p4est:   ${DEAL_II_WITH_P4EST}                                         ║")
message(STATUS "║    With UMFPACK: ${DEAL_II_WITH_UMFPACK}                                         ║")
message(STATUS "╠══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Options:                                                    ║")
message(STATUS "║    OpenMP:       ${ENABLE_OPENMP}                                          ║")
message(STATUS "║    Tests:        ${BUILD_TESTS}                                         ║")
message(STATUS "║    Benchmarks:   ${BUILD_BENCHMARKS}                                         ║")
message(STATUS "╚══════════════════════════════════════════════════════════════╝")
message(STATUS "")
