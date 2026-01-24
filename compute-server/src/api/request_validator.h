#ifndef REQUEST_VALIDATOR_H
#define REQUEST_VALIDATOR_H

/**
 * @file request_validator.h
 * @brief Request validation for FEA API
 * 
 * Validates incoming JSON requests for completeness and correctness
 * before processing.
 */

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace FEA {

using json = nlohmann::json;

/**
 * @brief Validation result with errors
 */
struct ValidationResult {
    bool valid;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    
    operator bool() const { return valid; }
    
    void add_error(const std::string& error) {
        valid = false;
        errors.push_back(error);
    }
    
    void add_warning(const std::string& warning) {
        warnings.push_back(warning);
    }
    
    std::string get_error_string() const {
        std::string result;
        for (size_t i = 0; i < errors.size(); ++i) {
            if (i > 0) result += "; ";
            result += errors[i];
        }
        return result;
    }
    
    json to_json() const {
        return {
            {"valid", valid},
            {"errors", errors},
            {"warnings", warnings}
        };
    }
};

/**
 * @brief Request validator for FEA analysis requests
 */
class RequestValidator {
public:
    /**
     * @brief Validate complete analysis request
     */
    static ValidationResult validate_analysis_request(const json& request);
    
    /**
     * @brief Validate mesh specification
     */
    static ValidationResult validate_mesh(const json& mesh);
    
    /**
     * @brief Validate boundary conditions array
     */
    static ValidationResult validate_boundary_conditions(const json& bcs);
    
    /**
     * @brief Validate single boundary condition
     */
    static ValidationResult validate_boundary_condition(const json& bc);
    
    /**
     * @brief Validate loads array
     */
    static ValidationResult validate_loads(const json& loads);
    
    /**
     * @brief Validate single load
     */
    static ValidationResult validate_load(const json& load);
    
    /**
     * @brief Validate connections array
     */
    static ValidationResult validate_connections(const json& connections);
    
    /**
     * @brief Validate materials specification
     */
    static ValidationResult validate_materials(const json& materials);
    
    /**
     * @brief Validate solver options
     */
    static ValidationResult validate_solver_options(const json& options);
    
    /**
     * @brief Validate unit system specification
     */
    static ValidationResult validate_units(const json& units);
    
    /**
     * @brief Validate boundary target specification
     */
    static ValidationResult validate_target(const json& target);
    
private:
    static bool is_valid_3d_point(const json& point);
    static bool is_valid_3d_vector(const json& vec);
    static bool is_positive(double value);
    static bool is_in_range(double value, double min, double max);
};

} // namespace FEA

#endif // REQUEST_VALIDATOR_H
