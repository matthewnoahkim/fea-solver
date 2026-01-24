#include "mesh_quality.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace FEA {

// ============================================================================
// Constructor
// ============================================================================

template <int dim>
MeshQualityAnalyzer<dim>::MeshQualityAnalyzer(
    const Triangulation<dim>& tria,
    const Mapping<dim>& map,
    const QualityThresholds& thresh)
    : triangulation(tria)
    , mapping(map)
    , thresholds(thresh)
    , analyzed(false)
{}

// ============================================================================
// Main Analysis Method
// ============================================================================

template <int dim>
MeshQualitySummary MeshQualityAnalyzer<dim>::analyze() {
    element_qualities.clear();
    element_qualities.reserve(triangulation.n_active_cells());
    
    // Initialize summary
    summary = MeshQualitySummary();
    summary.num_elements = triangulation.n_active_cells();
    summary.num_nodes = triangulation.n_vertices();
    
    summary.min_jacobian_ratio = std::numeric_limits<double>::max();
    summary.max_jacobian_ratio = 0;
    summary.min_aspect_ratio = std::numeric_limits<double>::max();
    summary.max_aspect_ratio = 0;
    summary.max_skewness = 0;
    summary.max_warpage = 0;
    summary.num_poor_quality = 0;
    
    double sum_jacobian = 0, sum_aspect = 0, sum_skewness = 0;
    
    unsigned int cell_index = 0;
    for (const auto& cell : triangulation.active_cell_iterators()) {
        ElementQuality eq;
        eq.cell_index = cell_index;
        eq.jacobian_ratio = compute_jacobian_ratio(cell);
        eq.aspect_ratio = compute_aspect_ratio(cell);
        eq.skewness = compute_skewness(cell);
        eq.warpage = (dim == 3) ? compute_warpage(cell) : 0.0;
        
        element_qualities.push_back(eq);
        
        // Update statistics
        summary.min_jacobian_ratio = std::min(summary.min_jacobian_ratio, eq.jacobian_ratio);
        summary.max_jacobian_ratio = std::max(summary.max_jacobian_ratio, eq.jacobian_ratio);
        summary.min_aspect_ratio = std::min(summary.min_aspect_ratio, eq.aspect_ratio);
        summary.max_aspect_ratio = std::max(summary.max_aspect_ratio, eq.aspect_ratio);
        summary.max_skewness = std::max(summary.max_skewness, eq.skewness);
        summary.max_warpage = std::max(summary.max_warpage, eq.warpage);
        
        sum_jacobian += eq.jacobian_ratio;
        sum_aspect += eq.aspect_ratio;
        sum_skewness += eq.skewness;
        
        // Track poor quality elements
        if (!eq.is_acceptable(thresholds)) {
            summary.num_poor_quality++;
            summary.poor_quality_cell_ids.push_back(cell_index);
        }
        
        ++cell_index;
    }
    
    // Compute averages
    if (summary.num_elements > 0) {
        summary.avg_jacobian_ratio = sum_jacobian / summary.num_elements;
        summary.avg_aspect_ratio = sum_aspect / summary.num_elements;
        summary.avg_skewness = sum_skewness / summary.num_elements;
    }
    
    // Generate warnings and errors based on analysis
    summary.is_acceptable = true;
    
    // Jacobian ratio checks
    if (summary.min_jacobian_ratio < thresholds.min_jacobian_ratio) {
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(4);
        msg << "Low Jacobian ratio detected (" << summary.min_jacobian_ratio
            << "). Elements may be highly distorted.";
        summary.warnings.push_back(msg.str());
        
        if (summary.min_jacobian_ratio < 0.01) {
            msg.str("");
            msg << "Jacobian ratio near zero (" << summary.min_jacobian_ratio
                << ") - mesh may be invalid or have inverted elements";
            summary.errors.push_back(msg.str());
            summary.is_acceptable = false;
        }
        
        if (summary.min_jacobian_ratio <= 0) {
            summary.errors.push_back("Negative Jacobian detected - mesh has inverted elements!");
            summary.is_acceptable = false;
        }
    }
    
    // Aspect ratio checks
    if (summary.max_aspect_ratio > thresholds.max_aspect_ratio) {
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(2);
        msg << "High aspect ratio detected (" << summary.max_aspect_ratio
            << "). Results may be less accurate in elongated elements.";
        summary.warnings.push_back(msg.str());
        
        if (summary.max_aspect_ratio > 100.0) {
            msg.str("");
            msg << "Extreme aspect ratio (" << summary.max_aspect_ratio
                << ") may cause solver convergence issues";
            summary.errors.push_back(msg.str());
            summary.is_acceptable = false;
        }
    }
    
    // Skewness checks
    if (summary.max_skewness > thresholds.max_skewness) {
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(3);
        msg << "High skewness detected (" << summary.max_skewness
            << "). Consider remeshing highly skewed regions.";
        summary.warnings.push_back(msg.str());
    }
    
    // Warpage checks (3D only)
    if (dim == 3 && summary.max_warpage > thresholds.max_warpage) {
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(3);
        msg << "High face warpage detected (" << summary.max_warpage
            << "). This can cause integration errors.";
        summary.warnings.push_back(msg.str());
    }
    
    // Overall poor quality percentage check
    double poor_percent = 100.0 * summary.num_poor_quality / 
                          std::max(1u, summary.num_elements);
    if (poor_percent > 10.0) {
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(1);
        msg << "More than 10% of elements (" << poor_percent
            << "%) have poor quality. Consider remeshing.";
        summary.errors.push_back(msg.str());
        summary.is_acceptable = false;
    } else if (poor_percent > 5.0) {
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(1);
        msg << poor_percent << "% of elements have poor quality.";
        summary.warnings.push_back(msg.str());
    } else if (poor_percent > 1.0) {
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(2);
        msg << poor_percent << "% of elements have marginal quality.";
        summary.warnings.push_back(msg.str());
    }
    
    analyzed = true;
    return summary;
}

// ============================================================================
// Element Quality Computation
// ============================================================================

template <int dim>
ElementQuality MeshQualityAnalyzer<dim>::get_element_quality(
    const typename Triangulation<dim>::active_cell_iterator& cell) const {
    
    ElementQuality eq;
    eq.cell_index = cell->index();
    eq.jacobian_ratio = compute_jacobian_ratio(cell);
    eq.aspect_ratio = compute_aspect_ratio(cell);
    eq.skewness = compute_skewness(cell);
    eq.warpage = (dim == 3) ? compute_warpage(cell) : 0.0;
    return eq;
}

// ============================================================================
// Jacobian Ratio Computation
// ============================================================================

template <int dim>
double MeshQualityAnalyzer<dim>::compute_jacobian_ratio(
    const typename Triangulation<dim>::active_cell_iterator& cell) const {
    
    // Use higher-order quadrature to capture Jacobian variation
    QGauss<dim> quadrature(3);
    FE_Q<dim> fe_dummy(1);
    FEValues<dim> fe_values(mapping, fe_dummy, quadrature, update_jacobians);
    
    fe_values.reinit(cell);
    
    double min_det = std::numeric_limits<double>::max();
    double max_det = 0;
    bool has_negative = false;
    
    for (unsigned int q = 0; q < quadrature.size(); ++q) {
        double det = determinant(fe_values.jacobian(q));
        
        if (det <= 0) {
            has_negative = true;
        }
        
        double abs_det = std::abs(det);
        min_det = std::min(min_det, abs_det);
        max_det = std::max(max_det, abs_det);
    }
    
    // Return negative ratio if element is inverted
    if (has_negative) {
        return -std::abs(min_det / std::max(max_det, 1e-14));
    }
    
    return (max_det > 1e-14) ? min_det / max_det : 0.0;
}

// ============================================================================
// Aspect Ratio Computation
// ============================================================================

template <int dim>
double MeshQualityAnalyzer<dim>::compute_aspect_ratio(
    const typename Triangulation<dim>::active_cell_iterator& cell) const {
    
    auto edge_lengths = get_edge_lengths(cell);
    
    if (edge_lengths.empty()) return 1.0;
    
    double min_len = *std::min_element(edge_lengths.begin(), edge_lengths.end());
    double max_len = *std::max_element(edge_lengths.begin(), edge_lengths.end());
    
    return (min_len > 1e-14) ? max_len / min_len : std::numeric_limits<double>::max();
}

template <int dim>
std::vector<double> MeshQualityAnalyzer<dim>::get_edge_lengths(
    const typename Triangulation<dim>::active_cell_iterator& cell) const {
    
    std::vector<double> lengths;
    lengths.reserve(GeometryInfo<dim>::lines_per_cell);
    
    for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line) {
        Point<dim> p0 = cell->line(line)->vertex(0);
        Point<dim> p1 = cell->line(line)->vertex(1);
        lengths.push_back(p0.distance(p1));
    }
    
    return lengths;
}

// ============================================================================
// Skewness Computation
// ============================================================================

template <int dim>
double MeshQualityAnalyzer<dim>::compute_skewness(
    const typename Triangulation<dim>::active_cell_iterator& cell) const {
    
    // Skewness based on volume ratio:
    // Compares actual volume to the volume of an ideal element with same average edge length
    
    double actual_volume = cell->measure();
    
    auto edge_lengths = get_edge_lengths(cell);
    if (edge_lengths.empty()) return 0.0;
    
    double avg_edge = std::accumulate(edge_lengths.begin(), edge_lengths.end(), 0.0)
                      / edge_lengths.size();
    
    // Compute ideal volume for regular element with average edge length
    double ideal_volume;
    if constexpr (dim == 3) {
        // For hexahedron: V = h^3
        ideal_volume = std::pow(avg_edge, 3);
    } else if constexpr (dim == 2) {
        // For quadrilateral: A = h^2
        ideal_volume = std::pow(avg_edge, 2);
    } else {
        ideal_volume = avg_edge;
    }
    
    if (ideal_volume < 1e-14) return 1.0;
    
    // Skewness = 1 - (actual/ideal), clamped to [0,1]
    double ratio = actual_volume / ideal_volume;
    
    // For highly distorted elements, ratio can be > 1 or very small
    // Normalize so perfect cube has skewness 0
    double skewness = std::abs(1.0 - ratio);
    return std::max(0.0, std::min(1.0, skewness));
}

template <int dim>
std::vector<double> MeshQualityAnalyzer<dim>::compute_face_angles(
    const typename Triangulation<dim>::active_cell_iterator& cell) const {
    
    std::vector<double> angles;
    
    // Compute angles at each vertex
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        Point<dim> vertex = cell->vertex(v);
        
        // Find edges meeting at this vertex
        std::vector<Tensor<1, dim>> edge_vectors;
        
        for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line) {
            auto line_obj = cell->line(line);
            if (line_obj->vertex(0) == vertex || line_obj->vertex(1) == vertex) {
                Tensor<1, dim> edge;
                Point<dim> other = (line_obj->vertex(0) == vertex) ? 
                                   line_obj->vertex(1) : line_obj->vertex(0);
                for (unsigned int d = 0; d < dim; ++d)
                    edge[d] = other[d] - vertex[d];
                if (edge.norm() > 1e-14) {
                    edge /= edge.norm();
                    edge_vectors.push_back(edge);
                }
            }
        }
        
        // Compute angles between pairs of edges
        for (size_t i = 0; i < edge_vectors.size(); ++i) {
            for (size_t j = i + 1; j < edge_vectors.size(); ++j) {
                double cos_angle = edge_vectors[i] * edge_vectors[j];
                cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
                angles.push_back(std::acos(cos_angle));
            }
        }
    }
    
    return angles;
}

// ============================================================================
// Warpage Computation (3D only)
// ============================================================================

template <int dim>
double MeshQualityAnalyzer<dim>::compute_warpage(
    const typename Triangulation<dim>::active_cell_iterator& cell) const {
    
    if constexpr (dim != 3) {
        return 0.0;
    }
    
    double max_warpage = 0;
    
    // Check each face for non-planarity
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
        // Get the 4 vertices of the face
        std::array<Point<dim>, 4> verts;
        for (unsigned int v = 0; v < 4; ++v) {
            verts[v] = cell->face(face)->vertex(v);
        }
        
        // Compute normal from first triangle (v0, v1, v2)
        Tensor<1, dim> v1, v2;
        for (unsigned int d = 0; d < dim; ++d) {
            v1[d] = verts[1][d] - verts[0][d];
            v2[d] = verts[2][d] - verts[0][d];
        }
        
        Tensor<1, dim> normal = cross_product_3d(v1, v2);
        double normal_mag = normal.norm();
        if (normal_mag < 1e-14) continue;
        normal /= normal_mag;
        
        // Compute distance from 4th vertex to the plane defined by first three
        Tensor<1, dim> v3;
        for (unsigned int d = 0; d < dim; ++d)
            v3[d] = verts[3][d] - verts[0][d];
        
        double dist = std::abs(v3 * normal);
        
        // Normalize by diagonal length
        double diag = verts[0].distance(verts[2]);
        double warpage = (diag > 1e-14) ? dist / diag : 0.0;
        
        max_warpage = std::max(max_warpage, warpage);
    }
    
    return max_warpage;
}

// ============================================================================
// Poor Quality Cell Identification
// ============================================================================

template <int dim>
std::vector<typename Triangulation<dim>::active_cell_iterator>
MeshQualityAnalyzer<dim>::get_poor_quality_cells() const {
    
    std::vector<typename Triangulation<dim>::active_cell_iterator> result;
    
    if (!analyzed) return result;
    
    unsigned int idx = 0;
    for (const auto& cell : triangulation.active_cell_iterators()) {
        if (idx < element_qualities.size() &&
            !element_qualities[idx].is_acceptable(thresholds)) {
            result.push_back(cell);
        }
        ++idx;
    }
    
    return result;
}

// ============================================================================
// Histogram Generation
// ============================================================================

template <int dim>
typename MeshQualityAnalyzer<dim>::HistogramData
MeshQualityAnalyzer<dim>::get_histogram(unsigned int num_bins) const {
    
    HistogramData data;
    
    if (!analyzed || element_qualities.empty()) return data;
    
    data.bin_edges.resize(num_bins + 1);
    data.jacobian_counts.resize(num_bins, 0);
    data.aspect_ratio_counts.resize(num_bins, 0);
    data.skewness_counts.resize(num_bins, 0);
    
    // Jacobian ratio bins: [0, 1]
    for (unsigned int i = 0; i <= num_bins; ++i) {
        data.bin_edges[i] = static_cast<double>(i) / num_bins;
    }
    
    // Aspect ratio range
    double ar_max = std::min(summary.max_aspect_ratio, 50.0);  // Cap for visualization
    
    for (const auto& eq : element_qualities) {
        // Jacobian histogram
        unsigned int jac_bin = static_cast<unsigned int>(
            std::max(0.0, eq.jacobian_ratio) * num_bins);
        jac_bin = std::min(jac_bin, num_bins - 1);
        data.jacobian_counts[jac_bin]++;
        
        // Aspect ratio histogram (normalized to [0,1] for binning)
        double ar_norm = std::min(1.0, (eq.aspect_ratio - 1.0) / (ar_max - 1.0));
        unsigned int ar_bin = static_cast<unsigned int>(ar_norm * num_bins);
        ar_bin = std::min(ar_bin, num_bins - 1);
        data.aspect_ratio_counts[ar_bin]++;
        
        // Skewness histogram
        unsigned int skew_bin = static_cast<unsigned int>(eq.skewness * num_bins);
        skew_bin = std::min(skew_bin, num_bins - 1);
        data.skewness_counts[skew_bin]++;
    }
    
    return data;
}

// ============================================================================
// Report Generation
// ============================================================================

template <int dim>
std::string MeshQualityAnalyzer<dim>::get_report() const {
    if (!analyzed) {
        return "Mesh quality not yet analyzed. Call analyze() first.";
    }
    
    std::ostringstream report;
    report << std::fixed << std::setprecision(4);
    
    report << "\n";
    report << "╔══════════════════════════════════════════════════════════════╗\n";
    report << "║                    MESH QUALITY REPORT                       ║\n";
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    report << "║  Elements: " << std::setw(10) << summary.num_elements
           << "     Nodes: " << std::setw(10) << summary.num_nodes << "       ║\n";
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    
    report << "║  JACOBIAN RATIO (higher is better, ideal = 1.0)              ║\n";
    report << "║    Min: " << std::setw(8) << summary.min_jacobian_ratio
           << "   Max: " << std::setw(8) << summary.max_jacobian_ratio
           << "   Avg: " << std::setw(8) << summary.avg_jacobian_ratio << "   ║\n";
    
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    report << "║  ASPECT RATIO (lower is better, ideal = 1.0)                 ║\n";
    report << "║    Min: " << std::setw(8) << summary.min_aspect_ratio
           << "   Max: " << std::setw(8) << summary.max_aspect_ratio
           << "   Avg: " << std::setw(8) << summary.avg_aspect_ratio << "   ║\n";
    
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    report << "║  SKEWNESS (lower is better, ideal = 0.0)                     ║\n";
    report << "║    Max: " << std::setw(8) << summary.max_skewness
           << "                    Avg: " << std::setw(8) << summary.avg_skewness << "   ║\n";
    
    if constexpr (dim == 3) {
        report << "╠══════════════════════════════════════════════════════════════╣\n";
        report << "║  WARPAGE (lower is better, ideal = 0.0)                      ║\n";
        report << "║    Max: " << std::setw(8) << summary.max_warpage
               << "                                          ║\n";
    }
    
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    double poor_pct = 100.0 * summary.num_poor_quality / 
                      std::max(1u, summary.num_elements);
    report << "║  Poor Quality Elements: " << std::setw(6) << summary.num_poor_quality
           << " (" << std::setw(5) << std::setprecision(1) << poor_pct << "%)                 ║\n";
    
    if (!summary.warnings.empty() || !summary.errors.empty()) {
        report << "╠══════════════════════════════════════════════════════════════╣\n";
    }
    
    if (!summary.warnings.empty()) {
        report << "║  WARNINGS:                                                   ║\n";
        for (const auto& w : summary.warnings) {
            std::string truncated = w.substr(0, 57);
            report << "║  ⚠ " << std::left << std::setw(58) << truncated << "║\n";
        }
    }
    
    if (!summary.errors.empty()) {
        report << "║  ERRORS:                                                     ║\n";
        for (const auto& e : summary.errors) {
            std::string truncated = e.substr(0, 57);
            report << "║  ✗ " << std::left << std::setw(58) << truncated << "║\n";
        }
    }
    
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    if (summary.is_acceptable) {
        report << "║  Overall Assessment: ✓ ACCEPTABLE                            ║\n";
    } else {
        report << "║  Overall Assessment: ✗ NEEDS IMPROVEMENT                     ║\n";
    }
    report << "╚══════════════════════════════════════════════════════════════╝\n";
    
    return report.str();
}

template <int dim>
json MeshQualityAnalyzer<dim>::to_json() const {
    if (!analyzed) {
        return {{"error", "Mesh quality not yet analyzed"}};
    }
    
    json j = summary.to_json();
    j["thresholds"] = thresholds.to_json();
    
    // Add histogram data
    auto hist = get_histogram(20);
    j["histogram"] = {
        {"bin_edges", hist.bin_edges},
        {"jacobian_counts", hist.jacobian_counts},
        {"aspect_ratio_counts", hist.aspect_ratio_counts},
        {"skewness_counts", hist.skewness_counts}
    };
    
    return j;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class MeshQualityAnalyzer<3>;
template class MeshQualityAnalyzer<2>;

} // namespace FEA
