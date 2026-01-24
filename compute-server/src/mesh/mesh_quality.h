#ifndef MESH_QUALITY_H
#define MESH_QUALITY_H

/**
 * @file mesh_quality.h
 * @brief Mesh quality analysis for FEA pre-processing
 * 
 * Provides comprehensive mesh quality metrics essential for ensuring
 * accurate finite element analysis results. Poor mesh quality leads to:
 * - Inaccurate stress calculations
 * - Solver convergence issues
 * - Artificial stress concentrations
 * 
 * Key metrics analyzed:
 * - Jacobian ratio: measures element distortion
 * - Aspect ratio: measures element elongation
 * - Skewness: measures deviation from ideal shape
 * - Warpage: measures face non-planarity (3D only)
 */

#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <limits>
#include <sstream>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

// ============================================================================
// Quality Thresholds
// ============================================================================

/**
 * @brief Configurable thresholds for mesh quality assessment
 */
struct QualityThresholds {
    double min_jacobian_ratio = 0.1;    ///< Minimum acceptable Jacobian ratio
    double max_aspect_ratio = 20.0;     ///< Maximum acceptable aspect ratio
    double max_skewness = 0.9;          ///< Maximum acceptable skewness
    double max_warpage = 0.5;           ///< Maximum acceptable warpage (3D)
    
    /**
     * @brief Strict thresholds for high-accuracy analyses
     */
    static QualityThresholds strict() {
        return {0.3, 10.0, 0.7, 0.3};
    }
    
    /**
     * @brief Standard thresholds for typical analyses
     */
    static QualityThresholds standard() {
        return {0.1, 20.0, 0.9, 0.5};
    }
    
    /**
     * @brief Relaxed thresholds for preliminary analyses
     */
    static QualityThresholds relaxed() {
        return {0.05, 50.0, 0.95, 0.7};
    }
    
    json to_json() const {
        return {
            {"min_jacobian_ratio", min_jacobian_ratio},
            {"max_aspect_ratio", max_aspect_ratio},
            {"max_skewness", max_skewness},
            {"max_warpage", max_warpage}
        };
    }
    
    static QualityThresholds from_json(const json& j) {
        QualityThresholds t;
        if (j.contains("min_jacobian_ratio")) t.min_jacobian_ratio = j["min_jacobian_ratio"];
        if (j.contains("max_aspect_ratio")) t.max_aspect_ratio = j["max_aspect_ratio"];
        if (j.contains("max_skewness")) t.max_skewness = j["max_skewness"];
        if (j.contains("max_warpage")) t.max_warpage = j["max_warpage"];
        return t;
    }
};

// ============================================================================
// Element Quality
// ============================================================================

/**
 * @brief Quality metrics for a single element
 */
struct ElementQuality {
    unsigned int cell_index;        ///< Element index in mesh
    double jacobian_ratio;          ///< Min/max Jacobian determinant ratio [0,1]
    double aspect_ratio;            ///< Max/min edge length ratio [1,∞)
    double skewness;                ///< Deviation from ideal shape [0,1]
    double warpage;                 ///< Face non-planarity (3D) [0,1]
    
    /**
     * @brief Check if element meets quality thresholds
     */
    bool is_acceptable(const QualityThresholds& thresholds) const {
        return jacobian_ratio >= thresholds.min_jacobian_ratio &&
               aspect_ratio <= thresholds.max_aspect_ratio &&
               skewness <= thresholds.max_skewness &&
               warpage <= thresholds.max_warpage;
    }
    
    /**
     * @brief Get overall quality score [0,1] where 1 is best
     */
    double overall_score() const {
        // Weighted average of normalized metrics
        double j_score = jacobian_ratio;  // Already in [0,1]
        double a_score = std::max(0.0, 1.0 - (aspect_ratio - 1.0) / 19.0);  // AR=1 -> 1, AR=20 -> 0
        double s_score = 1.0 - skewness;  // Skewness 0 -> 1, 1 -> 0
        double w_score = 1.0 - warpage;   // Warpage 0 -> 1, 1 -> 0
        
        return 0.4 * j_score + 0.3 * a_score + 0.2 * s_score + 0.1 * w_score;
    }
    
    json to_json() const {
        return {
            {"cell_index", cell_index},
            {"jacobian_ratio", jacobian_ratio},
            {"aspect_ratio", aspect_ratio},
            {"skewness", skewness},
            {"warpage", warpage},
            {"overall_score", overall_score()}
        };
    }
};

// ============================================================================
// Mesh Quality Summary
// ============================================================================

/**
 * @brief Summary statistics for entire mesh quality
 */
struct MeshQualitySummary {
    unsigned int num_elements;      ///< Total number of elements
    unsigned int num_nodes;         ///< Total number of nodes
    
    // Jacobian ratio statistics
    double min_jacobian_ratio;
    double max_jacobian_ratio;
    double avg_jacobian_ratio;
    
    // Aspect ratio statistics
    double min_aspect_ratio;
    double max_aspect_ratio;
    double avg_aspect_ratio;
    
    // Skewness statistics
    double max_skewness;
    double avg_skewness;
    
    // Warpage statistics (3D only)
    double max_warpage;
    
    // Poor quality tracking
    unsigned int num_poor_quality;
    std::vector<unsigned int> poor_quality_cell_ids;
    
    // Overall assessment
    bool is_acceptable;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    
    json to_json() const {
        return {
            {"num_elements", num_elements},
            {"num_nodes", num_nodes},
            {"metrics", {
                {"jacobian_ratio", {
                    {"min", min_jacobian_ratio},
                    {"max", max_jacobian_ratio},
                    {"avg", avg_jacobian_ratio}
                }},
                {"aspect_ratio", {
                    {"min", min_aspect_ratio},
                    {"max", max_aspect_ratio},
                    {"avg", avg_aspect_ratio}
                }},
                {"skewness", {
                    {"max", max_skewness},
                    {"avg", avg_skewness}
                }},
                {"warpage", {
                    {"max", max_warpage}
                }}
            }},
            {"poor_quality_elements", num_poor_quality},
            {"poor_quality_percent", 100.0 * num_poor_quality / std::max(1u, num_elements)},
            {"quality_acceptable", is_acceptable},
            {"warnings", warnings},
            {"errors", errors}
        };
    }
};

// ============================================================================
// Mesh Quality Analyzer
// ============================================================================

/**
 * @brief Analyzes mesh quality for FEA suitability
 * 
 * @tparam dim Spatial dimension (2 or 3)
 * 
 * Example usage:
 * @code
 * MeshQualityAnalyzer<3> analyzer(triangulation, mapping);
 * auto summary = analyzer.analyze();
 * std::cout << analyzer.get_report();
 * if (!summary.is_acceptable) {
 *     auto bad_cells = analyzer.get_poor_quality_cells();
 *     // Handle poor quality elements
 * }
 * @endcode
 */
template <int dim>
class MeshQualityAnalyzer {
public:
    /**
     * @brief Construct analyzer for a mesh
     * @param tria Triangulation to analyze
     * @param mapping Geometry mapping
     * @param thresholds Quality acceptance thresholds
     */
    MeshQualityAnalyzer(const Triangulation<dim>& tria,
                        const Mapping<dim>& mapping,
                        const QualityThresholds& thresholds = QualityThresholds());
    
    /**
     * @brief Analyze all elements and compute summary statistics
     */
    MeshQualitySummary analyze();
    
    /**
     * @brief Get quality metrics for a specific element
     */
    ElementQuality get_element_quality(
        const typename Triangulation<dim>::active_cell_iterator& cell) const;
    
    /**
     * @brief Get all elements that fail quality checks
     */
    std::vector<typename Triangulation<dim>::active_cell_iterator>
    get_poor_quality_cells() const;
    
    /**
     * @brief Get quality metric distributions for visualization
     */
    struct HistogramData {
        std::vector<double> bin_edges;
        std::vector<unsigned int> jacobian_counts;
        std::vector<unsigned int> aspect_ratio_counts;
        std::vector<unsigned int> skewness_counts;
    };
    HistogramData get_histogram(unsigned int num_bins = 20) const;
    
    /**
     * @brief Generate formatted quality report string
     */
    std::string get_report() const;
    
    /**
     * @brief Get analysis results as JSON
     */
    json to_json() const;
    
    /**
     * @brief Get all computed element qualities
     */
    const std::vector<ElementQuality>& get_element_qualities() const {
        return element_qualities;
    }
    
    /**
     * @brief Get the summary (must call analyze() first)
     */
    const MeshQualitySummary& get_summary() const {
        return summary;
    }
    
    /**
     * @brief Check if analysis has been performed
     */
    bool is_analyzed() const { return analyzed; }
    
private:
    const Triangulation<dim>& triangulation;
    const Mapping<dim>& mapping;
    QualityThresholds thresholds;
    
    mutable std::vector<ElementQuality> element_qualities;
    mutable MeshQualitySummary summary;
    mutable bool analyzed;
    
    /**
     * @brief Compute Jacobian ratio (min/max determinant at quadrature points)
     * 
     * Jacobian ratio measures element distortion. A ratio of 1.0 indicates
     * uniform mapping, while values near 0 indicate severe distortion.
     * Negative Jacobians indicate inverted elements.
     */
    double compute_jacobian_ratio(
        const typename Triangulation<dim>::active_cell_iterator& cell) const;
    
    /**
     * @brief Compute aspect ratio (max edge / min edge)
     * 
     * Aspect ratio measures element elongation. Highly elongated elements
     * can cause numerical issues, especially in directions perpendicular
     * to the long axis.
     */
    double compute_aspect_ratio(
        const typename Triangulation<dim>::active_cell_iterator& cell) const;
    
    /**
     * @brief Compute skewness (deviation from ideal shape)
     * 
     * Skewness measures how far an element deviates from its ideal shape.
     * A value of 0 indicates a perfectly regular element.
     */
    double compute_skewness(
        const typename Triangulation<dim>::active_cell_iterator& cell) const;
    
    /**
     * @brief Compute face warpage (3D only)
     * 
     * Warpage measures how non-planar the faces of a 3D element are.
     * High warpage can cause integration errors.
     */
    double compute_warpage(
        const typename Triangulation<dim>::active_cell_iterator& cell) const;
    
    /**
     * @brief Get all edge lengths for an element
     */
    std::vector<double> get_edge_lengths(
        const typename Triangulation<dim>::active_cell_iterator& cell) const;
    
    /**
     * @brief Compute angles at vertices for skewness calculation
     */
    std::vector<double> compute_face_angles(
        const typename Triangulation<dim>::active_cell_iterator& cell) const;
};

} // namespace FEA

#endif // MESH_QUALITY_H
