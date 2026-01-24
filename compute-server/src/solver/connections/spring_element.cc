#include "spring_element.h"
#include <numeric>

namespace FEA {

// ============================================================================
// Factory Function Implementations
// ============================================================================

SpringToGroundConnection create_nonlinear_spring_to_ground(
    const BoundaryTarget& target,
    std::function<double(double)> /*force_law*/,
    const std::string& description) {
    
    // Note: Nonlinear springs require iterative solution.
    // This is a placeholder that creates a linear spring.
    // Full nonlinear support would require:
    // 1. Storing the force law function
    // 2. Evaluating at each Newton-Raphson iteration
    // 3. Computing tangent stiffness from derivative
    
    SpringToGroundConnection conn;
    conn.target = target;
    conn.description = description;
    
    // Default to zero stiffness - must be set by caller
    conn.stiffness = Tensor<1, 3>();
    
    return conn;
}

std::vector<SpringConnection> create_spring_chain(
    const std::vector<Point<3>>& points,
    double stiffness) {
    
    std::vector<SpringConnection> springs;
    
    if (points.size() < 2) return springs;
    
    springs.reserve(points.size() - 1);
    
    for (size_t i = 0; i < points.size() - 1; ++i) {
        SpringConnection conn;
        conn.point_a = points[i];
        conn.point_b = points[i + 1];
        conn.axial_stiffness = stiffness;
        conn.description = "Chain spring " + std::to_string(i + 1);
        springs.push_back(conn);
    }
    
    return springs;
}

SpringConnection create_parallel_springs(
    const Point<3>& a, const Point<3>& b,
    const std::vector<double>& stiffnesses) {
    
    // Parallel springs: k_total = k1 + k2 + k3 + ...
    double total_k = std::accumulate(stiffnesses.begin(), stiffnesses.end(), 0.0);
    
    SpringConnection conn;
    conn.point_a = a;
    conn.point_b = b;
    conn.axial_stiffness = total_k;
    conn.description = "Parallel springs (n=" + std::to_string(stiffnesses.size()) + ")";
    
    return conn;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template FullMatrix<double> axial_spring_stiffness_global<3>(
    double k, const Tensor<1, 3>& axis);

} // namespace FEA
