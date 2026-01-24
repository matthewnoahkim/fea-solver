#include "request_validator.h"
#include <cmath>

namespace FEA {

// ============================================================================
// Helper Functions
// ============================================================================

bool RequestValidator::is_valid_3d_point(const json& point) {
    return point.is_array() && point.size() == 3 &&
           point[0].is_number() && point[1].is_number() && point[2].is_number();
}

bool RequestValidator::is_valid_3d_vector(const json& vec) {
    return is_valid_3d_point(vec);
}

bool RequestValidator::is_positive(double value) {
    return value > 0;
}

bool RequestValidator::is_in_range(double value, double min, double max) {
    return value >= min && value <= max;
}

// ============================================================================
// Main Validation
// ============================================================================

ValidationResult RequestValidator::validate_analysis_request(const json& request) {
    ValidationResult result{true, {}, {}};
    
    // Check required fields
    if (!request.contains("mesh")) {
        result.add_error("Missing required field: mesh");
    } else {
        auto mesh_result = validate_mesh(request["mesh"]);
        if (!mesh_result.valid) {
            for (const auto& err : mesh_result.errors)
                result.add_error("mesh: " + err);
        }
        for (const auto& warn : mesh_result.warnings)
            result.add_warning("mesh: " + warn);
    }
    
    if (!request.contains("boundary_conditions")) {
        result.add_error("Missing required field: boundary_conditions");
    } else {
        auto bc_result = validate_boundary_conditions(request["boundary_conditions"]);
        if (!bc_result.valid) {
            for (const auto& err : bc_result.errors)
                result.add_error("boundary_conditions: " + err);
        }
    }
    
    // Optional fields
    if (request.contains("loads")) {
        auto loads_result = validate_loads(request["loads"]);
        if (!loads_result.valid) {
            for (const auto& err : loads_result.errors)
                result.add_error("loads: " + err);
        }
    } else {
        result.add_warning("No loads specified - only boundary conditions will be applied");
    }
    
    if (request.contains("materials")) {
        auto mats_result = validate_materials(request["materials"]);
        if (!mats_result.valid) {
            for (const auto& err : mats_result.errors)
                result.add_error("materials: " + err);
        }
    }
    
    if (request.contains("solver_options")) {
        auto opts_result = validate_solver_options(request["solver_options"]);
        if (!opts_result.valid) {
            for (const auto& err : opts_result.errors)
                result.add_error("solver_options: " + err);
        }
    }
    
    if (request.contains("connections")) {
        auto conn_result = validate_connections(request["connections"]);
        if (!conn_result.valid) {
            for (const auto& err : conn_result.errors)
                result.add_error("connections: " + err);
        }
    }
    
    if (request.contains("units")) {
        auto units_result = validate_units(request["units"]);
        if (!units_result.valid) {
            for (const auto& err : units_result.errors)
                result.add_error("units: " + err);
        }
    }
    
    return result;
}

// ============================================================================
// Mesh Validation
// ============================================================================

ValidationResult RequestValidator::validate_mesh(const json& mesh) {
    ValidationResult result{true, {}, {}};
    
    if (!mesh.contains("type")) {
        result.add_error("Missing mesh type");
        return result;
    }
    
    std::string type = mesh["type"];
    
    if (type == "box") {
        if (!mesh.contains("min")) {
            result.add_error("box mesh requires 'min' point");
        } else if (!is_valid_3d_point(mesh["min"])) {
            result.add_error("'min' must be array of 3 numbers");
        }
        
        if (!mesh.contains("max")) {
            result.add_error("box mesh requires 'max' point");
        } else if (!is_valid_3d_point(mesh["max"])) {
            result.add_error("'max' must be array of 3 numbers");
        }
        
        // Check min < max
        if (result.valid && mesh.contains("min") && mesh.contains("max")) {
            auto min_pt = mesh["min"];
            auto max_pt = mesh["max"];
            for (int i = 0; i < 3; ++i) {
                if (min_pt[i].get<double>() >= max_pt[i].get<double>()) {
                    result.add_error("min[" + std::to_string(i) + "] must be less than max[" + 
                                    std::to_string(i) + "]");
                }
            }
        }
        
        if (mesh.contains("subdivisions")) {
            if (!mesh["subdivisions"].is_array() || mesh["subdivisions"].size() != 3) {
                result.add_error("subdivisions must be array of 3 integers");
            } else {
                for (int i = 0; i < 3; ++i) {
                    if (!mesh["subdivisions"][i].is_number_unsigned() ||
                        mesh["subdivisions"][i].get<unsigned int>() < 1) {
                        result.add_error("subdivisions must be positive integers");
                        break;
                    }
                }
            }
        }
    }
    else if (type == "cylinder") {
        if (!mesh.contains("radius")) {
            result.add_error("cylinder mesh requires 'radius'");
        } else if (!mesh["radius"].is_number() || mesh["radius"].get<double>() <= 0) {
            result.add_error("radius must be positive number");
        }
        
        if (!mesh.contains("height")) {
            result.add_error("cylinder mesh requires 'height'");
        } else if (!mesh["height"].is_number() || mesh["height"].get<double>() <= 0) {
            result.add_error("height must be positive number");
        }
    }
    else if (type == "file") {
        if (!mesh.contains("data") && !mesh.contains("path") && !mesh.contains("url")) {
            result.add_error("file mesh requires 'data', 'path', or 'url'");
        }
        
        if (mesh.contains("format")) {
            std::string format = mesh["format"];
            if (format != "msh" && format != "vtk" && format != "vtu" && 
                format != "inp" && format != "ucd" && format != "exo") {
                result.add_warning("Unknown mesh format '" + format + 
                                  "', supported: msh, vtk, vtu, inp, ucd, exo");
            }
        }
    }
    else {
        result.add_error("Unknown mesh type: " + type);
    }
    
    return result;
}

// ============================================================================
// Boundary Conditions Validation
// ============================================================================

ValidationResult RequestValidator::validate_boundary_conditions(const json& bcs) {
    ValidationResult result{true, {}, {}};
    
    if (!bcs.is_array()) {
        result.add_error("must be an array");
        return result;
    }
    
    if (bcs.empty()) {
        result.add_error("must not be empty (at least one BC required)");
        return result;
    }
    
    bool has_constraint = false;
    for (size_t i = 0; i < bcs.size(); ++i) {
        auto bc_result = validate_boundary_condition(bcs[i]);
        if (!bc_result.valid) {
            for (const auto& err : bc_result.errors)
                result.add_error("[" + std::to_string(i) + "]: " + err);
        }
        
        std::string type = bcs[i].value("type", "");
        if (type == "fixed" || type == "displacement" || type == "symmetry") {
            has_constraint = true;
        }
    }
    
    if (!has_constraint) {
        result.add_error("At least one displacement constraint (fixed, displacement, or symmetry) is required");
    }
    
    return result;
}

ValidationResult RequestValidator::validate_boundary_condition(const json& bc) {
    ValidationResult result{true, {}, {}};
    
    if (!bc.contains("type")) {
        result.add_error("missing 'type'");
        return result;
    }
    
    std::string type = bc["type"];
    
    if (type == "fixed") {
        // Target is required
        if (!bc.contains("target")) {
            result.add_error("fixed BC requires 'target'");
        } else {
            auto target_result = validate_target(bc["target"]);
            if (!target_result.valid) {
                for (const auto& err : target_result.errors)
                    result.add_error("target: " + err);
            }
        }
    }
    else if (type == "displacement") {
        if (!bc.contains("target")) {
            result.add_error("displacement BC requires 'target'");
        }
        if (!bc.contains("values")) {
            result.add_error("displacement BC requires 'values' array");
        } else if (!bc["values"].is_array() || bc["values"].size() != 3) {
            result.add_error("'values' must be array of 3 elements (use null for free)");
        }
    }
    else if (type == "symmetry") {
        if (!bc.contains("target")) {
            result.add_error("symmetry BC requires 'target'");
        }
        if (!bc.contains("plane_normal")) {
            result.add_error("symmetry BC requires 'plane_normal'");
        } else if (!is_valid_3d_vector(bc["plane_normal"])) {
            result.add_error("'plane_normal' must be array of 3 numbers");
        }
    }
    else if (type == "elastic_support") {
        if (!bc.contains("target")) {
            result.add_error("elastic_support BC requires 'target'");
        }
        if (!bc.contains("stiffness_per_area")) {
            result.add_error("elastic_support BC requires 'stiffness_per_area'");
        } else if (!is_valid_3d_vector(bc["stiffness_per_area"])) {
            result.add_error("'stiffness_per_area' must be array of 3 numbers");
        }
    }
    else if (type == "frictionless") {
        if (!bc.contains("target")) {
            result.add_error("frictionless BC requires 'target'");
        }
    }
    else {
        result.add_error("unknown type: " + type);
    }
    
    return result;
}

// ============================================================================
// Loads Validation
// ============================================================================

ValidationResult RequestValidator::validate_loads(const json& loads) {
    ValidationResult result{true, {}, {}};
    
    if (!loads.is_array()) {
        result.add_error("must be an array");
        return result;
    }
    
    for (size_t i = 0; i < loads.size(); ++i) {
        auto load_result = validate_load(loads[i]);
        if (!load_result.valid) {
            for (const auto& err : load_result.errors)
                result.add_error("[" + std::to_string(i) + "]: " + err);
        }
    }
    
    return result;
}

ValidationResult RequestValidator::validate_load(const json& load) {
    ValidationResult result{true, {}, {}};
    
    if (!load.contains("type")) {
        result.add_error("missing 'type'");
        return result;
    }
    
    std::string type = load["type"];
    
    if (type == "gravity") {
        if (!load.contains("acceleration")) {
            result.add_error("gravity load requires 'acceleration'");
        } else if (!is_valid_3d_vector(load["acceleration"])) {
            result.add_error("'acceleration' must be array of 3 numbers");
        }
    }
    else if (type == "pressure") {
        if (!load.contains("target")) {
            result.add_error("pressure load requires 'target'");
        }
        if (!load.contains("value")) {
            result.add_error("pressure load requires 'value'");
        } else if (!load["value"].is_number()) {
            result.add_error("'value' must be a number");
        }
    }
    else if (type == "surface_force") {
        if (!load.contains("target")) {
            result.add_error("surface_force load requires 'target'");
        }
        if (!load.contains("force_per_area")) {
            result.add_error("surface_force load requires 'force_per_area'");
        } else if (!is_valid_3d_vector(load["force_per_area"])) {
            result.add_error("'force_per_area' must be array of 3 numbers");
        }
    }
    else if (type == "point_force") {
        if (!load.contains("location")) {
            result.add_error("point_force load requires 'location'");
        } else if (!is_valid_3d_point(load["location"])) {
            result.add_error("'location' must be array of 3 numbers");
        }
        if (!load.contains("force")) {
            result.add_error("point_force load requires 'force'");
        } else if (!is_valid_3d_vector(load["force"])) {
            result.add_error("'force' must be array of 3 numbers");
        }
    }
    else if (type == "moment") {
        if (!load.contains("location")) {
            result.add_error("moment load requires 'location'");
        }
        if (!load.contains("moment")) {
            result.add_error("moment load requires 'moment' vector");
        } else if (!is_valid_3d_vector(load["moment"])) {
            result.add_error("'moment' must be array of 3 numbers");
        }
    }
    else if (type == "thermal") {
        if (!load.contains("reference_temperature")) {
            result.add_error("thermal load requires 'reference_temperature'");
        }
        if (!load.contains("applied_temperature")) {
            result.add_error("thermal load requires 'applied_temperature'");
        }
    }
    else if (type == "centrifugal") {
        if (!load.contains("axis")) {
            result.add_error("centrifugal load requires 'axis'");
        } else if (!is_valid_3d_vector(load["axis"])) {
            result.add_error("'axis' must be array of 3 numbers");
        }
        if (!load.contains("angular_velocity")) {
            result.add_error("centrifugal load requires 'angular_velocity'");
        }
    }
    else {
        result.add_warning("unknown load type: " + type);
    }
    
    return result;
}

// ============================================================================
// Connections Validation
// ============================================================================

ValidationResult RequestValidator::validate_connections(const json& connections) {
    ValidationResult result{true, {}, {}};
    
    if (!connections.is_array()) {
        result.add_error("must be an array");
        return result;
    }
    
    for (size_t i = 0; i < connections.size(); ++i) {
        const auto& conn = connections[i];
        if (!conn.contains("type")) {
            result.add_error("[" + std::to_string(i) + "]: missing 'type'");
            continue;
        }
        
        std::string type = conn["type"];
        if (type == "spring_to_ground") {
            if (!conn.contains("location"))
                result.add_error("[" + std::to_string(i) + "]: spring_to_ground requires 'location'");
            if (!conn.contains("stiffness"))
                result.add_error("[" + std::to_string(i) + "]: spring_to_ground requires 'stiffness'");
        }
        else if (type == "rigid" || type == "rbe2") {
            if (!conn.contains("master"))
                result.add_error("[" + std::to_string(i) + "]: rigid connection requires 'master'");
            if (!conn.contains("slave"))
                result.add_error("[" + std::to_string(i) + "]: rigid connection requires 'slave'");
        }
    }
    
    return result;
}

// ============================================================================
// Materials Validation
// ============================================================================

ValidationResult RequestValidator::validate_materials(const json& materials) {
    ValidationResult result{true, {}, {}};
    
    if (materials.contains("custom")) {
        if (!materials["custom"].is_array()) {
            result.add_error("'custom' must be an array");
            return result;
        }
        
        for (size_t i = 0; i < materials["custom"].size(); ++i) {
            const auto& mat = materials["custom"][i];
            if (!mat.contains("name")) {
                result.add_error("custom[" + std::to_string(i) + "]: missing 'name'");
            }
            
            // Check for required material properties based on model
            std::string model = mat.value("model", "isotropic_elastic");
            if (model == "isotropic_elastic") {
                if (!mat.contains("youngs_modulus"))
                    result.add_error("custom[" + std::to_string(i) + "]: missing 'youngs_modulus'");
                if (!mat.contains("poissons_ratio"))
                    result.add_error("custom[" + std::to_string(i) + "]: missing 'poissons_ratio'");
                
                if (mat.contains("poissons_ratio")) {
                    double nu = mat["poissons_ratio"];
                    if (nu <= -1.0 || nu >= 0.5) {
                        result.add_error("custom[" + std::to_string(i) + 
                                        "]: poissons_ratio must be in range (-1, 0.5)");
                    }
                }
            }
        }
    }
    
    return result;
}

// ============================================================================
// Solver Options Validation
// ============================================================================

ValidationResult RequestValidator::validate_solver_options(const json& options) {
    ValidationResult result{true, {}, {}};
    
    if (options.contains("fe_degree")) {
        int degree = options["fe_degree"];
        if (degree < 1 || degree > 4) {
            result.add_warning("fe_degree outside typical range [1,4]");
        }
    }
    
    if (options.contains("refinement_cycles")) {
        int cycles = options["refinement_cycles"];
        if (cycles < 0) {
            result.add_error("refinement_cycles must be non-negative");
        } else if (cycles > 10) {
            result.add_warning("refinement_cycles > 10 may be slow");
        }
    }
    
    if (options.contains("tolerance")) {
        double tol = options["tolerance"];
        if (tol <= 0) {
            result.add_error("tolerance must be positive");
        } else if (tol > 0.01) {
            result.add_warning("tolerance > 0.01 may give inaccurate results");
        }
    }
    
    if (options.contains("max_iterations")) {
        int max_iter = options["max_iterations"];
        if (max_iter < 1) {
            result.add_error("max_iterations must be at least 1");
        }
    }
    
    return result;
}

// ============================================================================
// Units Validation
// ============================================================================

ValidationResult RequestValidator::validate_units(const json& units) {
    ValidationResult result{true, {}, {}};
    
    if (!units.contains("type")) {
        result.add_error("missing 'type'");
        return result;
    }
    
    std::string type = units["type"];
    if (type != "SI" && type != "SI_MM" && type != "US_CUSTOMARY" && type != "custom") {
        result.add_error("unknown unit system: " + type + 
                        " (supported: SI, SI_MM, US_CUSTOMARY, custom)");
    }
    
    if (type == "custom") {
        if (!units.contains("factors")) {
            result.add_error("custom units require 'factors'");
        }
    }
    
    return result;
}

// ============================================================================
// Target Validation
// ============================================================================

ValidationResult RequestValidator::validate_target(const json& target) {
    ValidationResult result{true, {}, {}};
    
    if (!target.contains("type")) {
        result.add_error("missing 'type'");
        return result;
    }
    
    std::string type = target["type"];
    
    if (type == "boundary_id") {
        if (!target.contains("id")) {
            result.add_error("boundary_id target requires 'id'");
        } else if (!target["id"].is_number_unsigned()) {
            result.add_error("'id' must be a non-negative integer");
        }
    }
    else if (type == "point") {
        if (!target.contains("location")) {
            result.add_error("point target requires 'location'");
        } else if (!is_valid_3d_point(target["location"])) {
            result.add_error("'location' must be array of 3 numbers");
        }
    }
    else if (type == "box") {
        if (!target.contains("min") || !target.contains("max")) {
            result.add_error("box target requires 'min' and 'max'");
        }
    }
    else if (type == "plane") {
        if (!target.contains("point") || !target.contains("normal")) {
            result.add_error("plane target requires 'point' and 'normal'");
        }
    }
    else {
        result.add_error("unknown target type: " + type);
    }
    
    return result;
}

} // namespace FEA
