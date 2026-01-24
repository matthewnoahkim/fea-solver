#ifndef JSON_PARSER_H
#define JSON_PARSER_H

/**
 * @file json_parser.h
 * @brief JSON parsing utilities for FEA API
 * 
 * Provides helper functions for parsing JSON data into FEA types
 * with proper validation and error handling.
 */

#include <nlohmann/json.hpp>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <optional>
#include <string>
#include <vector>

namespace FEA {

using json = nlohmann::json;
using namespace dealii;

/**
 * @brief JSON parsing utilities
 */
class JsonParser {
public:
    // =========================================================================
    // Point/Tensor Parsing
    // =========================================================================
    
    /**
     * @brief Parse a Point<dim> from JSON array
     */
    template <int dim>
    static Point<dim> parse_point(const json& j) {
        auto arr = j.get<std::vector<double>>();
        Point<dim> p;
        for (unsigned int d = 0; d < dim && d < arr.size(); ++d)
            p[d] = arr[d];
        return p;
    }
    
    /**
     * @brief Parse a Tensor<1,dim> from JSON array
     */
    template <int dim>
    static Tensor<1, dim> parse_vector(const json& j) {
        auto arr = j.get<std::vector<double>>();
        Tensor<1, dim> t;
        for (unsigned int d = 0; d < dim && d < arr.size(); ++d)
            t[d] = arr[d];
        return t;
    }
    
    /**
     * @brief Parse optional value
     */
    template <typename T>
    static std::optional<T> parse_optional(const json& j, const std::string& key) {
        if (j.contains(key) && !j[key].is_null()) {
            return j[key].get<T>();
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get value with default
     */
    template <typename T>
    static T get_or_default(const json& j, const std::string& key, const T& default_value) {
        if (j.contains(key) && !j[key].is_null()) {
            return j[key].get<T>();
        }
        return default_value;
    }
    
    // =========================================================================
    // Array Parsing
    // =========================================================================
    
    /**
     * @brief Parse array of nullable values
     */
    template <typename T, size_t N>
    static std::array<std::optional<T>, N> parse_optional_array(const json& j) {
        std::array<std::optional<T>, N> result;
        auto arr = j.get<std::vector<json>>();
        for (size_t i = 0; i < N && i < arr.size(); ++i) {
            if (!arr[i].is_null()) {
                result[i] = arr[i].get<T>();
            }
        }
        return result;
    }
    
    // =========================================================================
    // JSON Conversion
    // =========================================================================
    
    /**
     * @brief Convert Point to JSON array
     */
    template <int dim>
    static json point_to_json(const Point<dim>& p) {
        std::vector<double> arr(dim);
        for (unsigned int d = 0; d < dim; ++d)
            arr[d] = p[d];
        return arr;
    }
    
    /**
     * @brief Convert Tensor to JSON array
     */
    template <int dim>
    static json tensor_to_json(const Tensor<1, dim>& t) {
        std::vector<double> arr(dim);
        for (unsigned int d = 0; d < dim; ++d)
            arr[d] = t[d];
        return arr;
    }
    
    /**
     * @brief Convert SymmetricTensor to JSON
     */
    template <int dim>
    static json symmetric_tensor_to_json(const SymmetricTensor<2, dim>& t) {
        json j;
        if constexpr (dim == 3) {
            j["xx"] = t[0][0];
            j["yy"] = t[1][1];
            j["zz"] = t[2][2];
            j["xy"] = t[0][1];
            j["xz"] = t[0][2];
            j["yz"] = t[1][2];
        } else {
            j["xx"] = t[0][0];
            j["yy"] = t[1][1];
            j["xy"] = t[0][1];
        }
        return j;
    }
    
    // =========================================================================
    // Validation Helpers
    // =========================================================================
    
    /**
     * @brief Check if field exists and is array of expected size
     */
    static bool validate_array(const json& j, const std::string& key, 
                               size_t expected_size, std::string& error) {
        if (!j.contains(key)) {
            error = "Missing required field: " + key;
            return false;
        }
        if (!j[key].is_array()) {
            error = key + " must be an array";
            return false;
        }
        if (j[key].size() != expected_size) {
            error = key + " must have " + std::to_string(expected_size) + " elements";
            return false;
        }
        return true;
    }
    
    /**
     * @brief Check if field exists and is numeric
     */
    static bool validate_number(const json& j, const std::string& key,
                                std::string& error) {
        if (!j.contains(key)) {
            error = "Missing required field: " + key;
            return false;
        }
        if (!j[key].is_number()) {
            error = key + " must be a number";
            return false;
        }
        return true;
    }
    
    /**
     * @brief Check if field exists and is string
     */
    static bool validate_string(const json& j, const std::string& key,
                                std::string& error) {
        if (!j.contains(key)) {
            error = "Missing required field: " + key;
            return false;
        }
        if (!j[key].is_string()) {
            error = key + " must be a string";
            return false;
        }
        return true;
    }
};

} // namespace FEA

#endif // JSON_PARSER_H
