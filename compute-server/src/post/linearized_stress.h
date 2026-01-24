#ifndef LINEARIZED_STRESS_H
#define LINEARIZED_STRESS_H

/**
 * @file linearized_stress.h
 * @brief Linearized stress analysis per ASME BPVC Section VIII, Division 2
 * 
 * Implements stress linearization along Stress Classification Lines (SCLs)
 * for pressure vessel and piping analysis. Decomposes stress into:
 * - Membrane stress (Pm): Average through thickness
 * - Bending stress (Pb): Linear variation through thickness
 * - Peak stress (F): Remaining non-linear component
 * 
 * Used for ASME code compliance checking with allowable stress intensities.
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/symmetric_tensor.h>

#include "../solver/material_library.h"

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <map>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

/**
 * @brief Linearized stress calculator per ASME BPVC
 * 
 * @tparam dim Spatial dimension (2 or 3)
 */
template <int dim>
class LinearizedStressCalculator {
public:
    /**
     * @brief Definition of a Stress Classification Line
     */
    struct StressClassificationLine {
        Point<dim> start;           ///< Inner surface point
        Point<dim> end;             ///< Outer surface point
        std::string name;           ///< SCL identifier
        unsigned int num_points = 20; ///< Integration points along line
        
        json to_json() const {
            return {
                {"name", name},
                {"start", std::vector<double>{start[0], start[1], 
                    dim == 3 ? start[2] : 0.0}},
                {"end", std::vector<double>{end[0], end[1],
                    dim == 3 ? end[2] : 0.0}},
                {"num_points", num_points}
            };
        }
    };
    
    /**
     * @brief Results of linearization for one SCL
     */
    struct LinearizedResult {
        std::string scl_name;
        Point<dim> start;
        Point<dim> end;
        double thickness;               ///< Length of SCL
        
        // Von Mises equivalent stresses
        double membrane;                ///< Pm - General primary membrane
        double bending;                 ///< Pb - Primary bending
        double peak;                    ///< F - Peak stress
        double membrane_plus_bending;   ///< Pm + Pb
        double total;                   ///< Pm + Pb + F (at surface)
        
        // Stress intensity (Tresca-based per ASME)
        double membrane_intensity;              ///< SI of membrane
        double membrane_plus_bending_intensity; ///< SI of Pm+Pb
        
        // Full tensor components
        SymmetricTensor<2, dim> membrane_tensor;
        SymmetricTensor<2, dim> bending_tensor;
        
        // ASME allowables
        double Sm;                              ///< Allowable stress intensity
        double membrane_utilization;            ///< Pm / Sm
        double membrane_plus_bending_utilization; ///< (Pm+Pb) / (1.5*Sm)
        bool membrane_ok;                       ///< Pm <= Sm
        bool membrane_plus_bending_ok;          ///< Pm+Pb <= 1.5*Sm
        
        json to_json() const {
            json j;
            j["scl_name"] = scl_name;
            j["start"] = std::vector<double>{start[0], start[1],
                dim == 3 ? start[2] : 0.0};
            j["end"] = std::vector<double>{end[0], end[1],
                dim == 3 ? end[2] : 0.0};
            j["thickness"] = thickness;
            j["membrane"] = membrane;
            j["bending"] = bending;
            j["peak"] = peak;
            j["membrane_plus_bending"] = membrane_plus_bending;
            j["total"] = total;
            j["membrane_intensity"] = membrane_intensity;
            j["membrane_plus_bending_intensity"] = membrane_plus_bending_intensity;
            j["Sm"] = Sm;
            j["membrane_utilization"] = membrane_utilization;
            j["membrane_plus_bending_utilization"] = membrane_plus_bending_utilization;
            j["membrane_ok"] = membrane_ok;
            j["membrane_plus_bending_ok"] = membrane_plus_bending_ok;
            return j;
        }
    };
    
    /**
     * @brief Construct calculator
     */
    LinearizedStressCalculator(const DoFHandler<dim>& dof_handler,
                               const Mapping<dim>& mapping,
                               const std::map<unsigned int, Material>& materials);
    
    // =========================================================================
    // SCL Management
    // =========================================================================
    
    /**
     * @brief Add SCL by endpoints
     */
    void add_scl(const Point<dim>& start, const Point<dim>& end,
                 const std::string& name = "",
                 unsigned int num_points = 20);
    
    /**
     * @brief Add SCL from struct
     */
    void add_scl(const StressClassificationLine& scl);
    
    /**
     * @brief Add multiple SCLs at once
     */
    void add_scls(const std::vector<StressClassificationLine>& scls);
    
    /**
     * @brief Clear all SCLs
     */
    void clear_scls();
    
    /**
     * @brief Get defined SCLs
     */
    const std::vector<StressClassificationLine>& get_scls() const { return scls_; }
    
    // =========================================================================
    // Computation
    // =========================================================================
    
    /**
     * @brief Compute linearized stresses for all SCLs
     * @param solution Displacement solution vector
     */
    void compute(const Vector<double>& solution);
    
    // =========================================================================
    // Results
    // =========================================================================
    
    /**
     * @brief Get all results
     */
    std::vector<LinearizedResult> get_results() const { return results_; }
    
    /**
     * @brief Get result for a specific SCL by name
     */
    LinearizedResult get_result(const std::string& name) const;
    
    /**
     * @brief Check if all SCLs pass ASME limits
     */
    bool all_pass() const;
    
    /**
     * @brief Get text report
     */
    std::string get_report() const;
    
    /**
     * @brief Get JSON results
     */
    json to_json() const;
    
private:
    const DoFHandler<dim>& dof_handler_;
    const Mapping<dim>& mapping_;
    const std::map<unsigned int, Material>& materials_;
    
    std::vector<StressClassificationLine> scls_;
    std::vector<LinearizedResult> results_;
    
    /**
     * @brief Compute linearization for one SCL
     */
    void compute_scl(const StressClassificationLine& scl,
                     const Vector<double>& solution);
    
    /**
     * @brief Get stress tensor at arbitrary point
     */
    SymmetricTensor<2, dim> get_stress_at_point(
        const Point<dim>& p,
        const Vector<double>& solution) const;
    
    /**
     * @brief Compute stress intensity (ASME Tresca-based)
     */
    double compute_stress_intensity(const SymmetricTensor<2, dim>& stress) const;
    
    /**
     * @brief Compute von Mises equivalent
     */
    double compute_von_mises(const SymmetricTensor<2, dim>& stress) const;
    
    /**
     * @brief Get ASME Sm allowable from material
     */
    double get_sm_allowable(unsigned int material_id) const;
};

} // namespace FEA

#endif // LINEARIZED_STRESS_H
