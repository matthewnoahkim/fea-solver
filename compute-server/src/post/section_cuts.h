#ifndef SECTION_CUTS_H
#define SECTION_CUTS_H

/**
 * @file section_cuts.h
 * @brief Section cut analysis for force and moment resultants
 * 
 * Computes section forces and moments by integrating stress over
 * a cutting plane through the model. Useful for:
 * - Beam resultants (axial, shear, moment)
 * - Free body diagram verification
 * - Load path analysis
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>

#include "../solver/material_library.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

/**
 * @brief Section cut analysis for structural loads
 * 
 * @tparam dim Spatial dimension (2 or 3)
 */
template <int dim>
class SectionCutAnalyzer {
public:
    /**
     * @brief Definition of a section cut plane
     */
    struct SectionCut {
        Point<dim> point;       ///< Point on the cutting plane
        Tensor<1, dim> normal;  ///< Normal direction (positive side)
        std::string name;       ///< Cut identifier
        
        json to_json() const {
            json j;
            j["name"] = name;
            j["point"] = std::vector<double>{point[0], point[1],
                dim == 3 ? point[2] : 0.0};
            j["normal"] = std::vector<double>{normal[0], normal[1],
                dim == 3 ? normal[2] : 0.0};
            return j;
        }
    };
    
    /**
     * @brief Results of section cut analysis
     */
    struct SectionResult {
        std::string name;
        Point<dim> centroid;        ///< Section centroid
        double area;                ///< Cross-sectional area
        
        // Force resultants in global coordinates
        Tensor<1, dim> force;       ///< Total force vector [N]
        Tensor<1, dim> moment;      ///< Total moment about centroid [N·m]
        
        // Force resultants in section local coordinates
        double axial_force;         ///< Force normal to section [N]
        Tensor<1, dim-1> shear_force; ///< In-plane shear forces [N]
        
        // For beams (about section centroid)
        double bending_moment_y;    ///< Bending about local Y [N·m]
        double bending_moment_z;    ///< Bending about local Z [N·m] (3D)
        double torsion;             ///< Twisting moment [N·m] (3D)
        
        json to_json() const {
            json j;
            j["name"] = name;
            j["centroid"] = std::vector<double>{centroid[0], centroid[1],
                dim == 3 ? centroid[2] : 0.0};
            j["area"] = area;
            j["force"] = std::vector<double>{force[0], force[1],
                dim == 3 ? force[2] : 0.0};
            j["moment"] = std::vector<double>{moment[0], moment[1],
                dim == 3 ? moment[2] : 0.0};
            j["axial_force"] = axial_force;
            if constexpr (dim == 3) {
                j["bending_moment_y"] = bending_moment_y;
                j["bending_moment_z"] = bending_moment_z;
                j["torsion"] = torsion;
            } else {
                j["bending_moment"] = bending_moment_y;
            }
            return j;
        }
    };
    
    /**
     * @brief Construct analyzer
     */
    SectionCutAnalyzer(const DoFHandler<dim>& dof_handler,
                       const Mapping<dim>& mapping,
                       const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Add a section cut definition
     */
    void add_cut(const Point<dim>& point,
                 const Tensor<1, dim>& normal,
                 const std::string& name = "");
    
    void add_cut(const SectionCut& cut);
    void clear_cuts();
    
    /**
     * @brief Compute section forces for all defined cuts
     */
    void compute(const Vector<double>& solution);
    
    /**
     * @brief Get results
     */
    std::vector<SectionResult> get_results() const { return results_; }
    SectionResult get_result(const std::string& name) const;
    
    std::string get_report() const;
    json to_json() const;
    
private:
    const DoFHandler<dim>& dof_handler_;
    const Mapping<dim>& mapping_;
    const std::map<unsigned int, Material>& materials_;
    
    std::vector<SectionCut> cuts_;
    std::vector<SectionResult> results_;
    
    void compute_cut(const SectionCut& cut, const Vector<double>& solution);
};

} // namespace FEA

#endif // SECTION_CUTS_H
