#ifndef SAFETY_FACTORS_H
#define SAFETY_FACTORS_H

/**
 * @file safety_factors.h
 * @brief Safety factor calculation against various failure criteria
 * 
 * Computes safety factors (factor of safety) by comparing computed
 * stresses against material allowable stresses for:
 * - Yield (plastic deformation)
 * - Ultimate (fracture)
 * - Fatigue (cyclic loading)
 * - Custom allowable values
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>

#include "../solver/material_library.h"

#include <nlohmann/json.hpp>
#include <map>
#include <vector>
#include <optional>
#include <string>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

/**
 * @brief Computes safety factors for structural analysis
 * 
 * @tparam dim Spatial dimension (2 or 3)
 */
template <int dim>
class SafetyFactorCalculator {
public:
    /**
     * @brief Failure criterion for safety factor calculation
     */
    enum class Criterion {
        VON_MISES_YIELD,        ///< σ_vm vs yield strength
        VON_MISES_ULTIMATE,     ///< σ_vm vs ultimate strength
        TRESCA_YIELD,           ///< τ_max vs yield/2
        TRESCA_ULTIMATE,        ///< τ_max vs ultimate/2
        MAX_PRINCIPAL_YIELD,    ///< σ_1 vs yield (brittle)
        MAX_PRINCIPAL_ULTIMATE, ///< σ_1 vs ultimate (brittle)
        GOODMAN_FATIGUE,        ///< Goodman diagram fatigue
        CUSTOM_ALLOWABLE        ///< User-defined allowable
    };
    
    /**
     * @brief Statistical summary of safety factors
     */
    struct SafetyFactorStatistics {
        double min_sf;                      ///< Minimum safety factor
        double max_sf;                      ///< Maximum safety factor
        double avg_sf;                      ///< Average safety factor
        double volume_weighted_min_sf;      ///< Min SF over worst 1% volume
        Point<dim> min_sf_location;         ///< Location of minimum SF
        unsigned int min_sf_material_id;    ///< Material at minimum SF
        double percent_below_1_0;           ///< % elements failing
        double percent_below_1_25;          ///< % elements near failure
        double percent_below_1_5;           ///< % elements with low margin
        double percent_below_2_0;           ///< % elements needing review
        double percent_below_3_0;           ///< % elements with moderate margin
        
        json to_json() const {
            return {
                {"min_sf", min_sf},
                {"max_sf", max_sf},
                {"avg_sf", avg_sf},
                {"volume_weighted_min_sf", volume_weighted_min_sf},
                {"min_sf_location", std::vector<double>{min_sf_location[0],
                    min_sf_location[1], dim == 3 ? min_sf_location[2] : 0.0}},
                {"distribution", {
                    {"below_1_0", percent_below_1_0},
                    {"below_1_25", percent_below_1_25},
                    {"below_1_5", percent_below_1_5},
                    {"below_2_0", percent_below_2_0},
                    {"below_3_0", percent_below_3_0}
                }}
            };
        }
    };
    
    /**
     * @brief Construct calculator
     */
    SafetyFactorCalculator(const DoFHandler<dim>& dof_handler,
                           const Mapping<dim>& mapping,
                           const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Set custom allowable stress for all materials
     */
    void set_custom_allowable(double allowable);
    
    /**
     * @brief Set design factor (divides allowable)
     */
    void set_design_factor(double factor) { design_factor_ = factor; }
    
    /**
     * @brief Compute safety factors from stress field
     * @param stress_field Cell-averaged stress values (e.g., von Mises)
     * @param criterion Failure criterion to use
     */
    void compute(const Vector<double>& stress_field,
                 Criterion criterion = Criterion::VON_MISES_YIELD);
    
    // =========================================================================
    // Result Accessors
    // =========================================================================
    
    double get_min_safety_factor() const { return stats_.min_sf; }
    Point<dim> get_min_sf_location() const { return stats_.min_sf_location; }
    const Vector<double>& get_safety_factor_field() const { return sf_field_; }
    SafetyFactorStatistics get_statistics() const { return stats_; }
    
    /**
     * @brief Get percentage of volume below a threshold SF
     */
    double get_percent_below_threshold(double threshold) const;
    
    /**
     * @brief Get volume-weighted minimum SF (over worst fraction)
     */
    double get_volume_weighted_min_sf(double volume_fraction = 0.01) const;
    
    /**
     * @brief Check if design meets required safety factor
     */
    bool passes_design_check(double required_sf = 1.0) const;
    
    /**
     * @brief Generate text assessment report
     */
    std::string get_assessment_report() const;
    
    /**
     * @brief Get results as JSON
     */
    json to_json() const;
    
    /**
     * @brief Get criterion name as string
     */
    static std::string criterion_to_string(Criterion c);
    
private:
    const DoFHandler<dim>& dof_handler_;
    const Mapping<dim>& mapping_;
    const std::map<unsigned int, Material>& materials_;
    
    Vector<double> sf_field_;
    SafetyFactorStatistics stats_;
    Criterion current_criterion_;
    std::optional<double> custom_allowable_;
    double design_factor_ = 1.0;
    
    /**
     * @brief Get allowable stress for a material and criterion
     */
    double get_allowable_stress(unsigned int material_id, Criterion criterion) const;
    
    /**
     * @brief Compute distribution statistics
     */
    void compute_statistics();
};

} // namespace FEA

#endif // SAFETY_FACTORS_H
