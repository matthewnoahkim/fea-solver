#include "rigid_constraint.h"
#include <deal.II/base/symmetric_tensor.h>
#include <algorithm>
#include <numeric>

namespace FEA {

// ============================================================================
// Rigid Body Motion Utilities
// ============================================================================

template <int dim>
FullMatrix<double> rigid_body_transformation_matrix(const Tensor<1, dim>& r) {
    // For 3D: T is 3x6 matrix
    // u = T * [ux, uy, uz, wx, wy, wz]^T
    // u = u_ref + omega × r
    
    FullMatrix<double> T(dim, 2 * dim);
    
    // Translation part (identity)
    for (unsigned int i = 0; i < dim; ++i) {
        T(i, i) = 1.0;
    }
    
    // Rotation part (skew-symmetric of r)
    if constexpr (dim == 3) {
        // omega × r = [wy*rz - wz*ry, wz*rx - wx*rz, wx*ry - wy*rx]
        // d(u)/d(omega) = [-r]× = [  0   rz  -ry ]
        //                         [ -rz   0   rx ]
        //                         [  ry  -rx   0 ]
        T(0, 3) = 0.0;    T(0, 4) = r[2];   T(0, 5) = -r[1];
        T(1, 3) = -r[2];  T(1, 4) = 0.0;    T(1, 5) = r[0];
        T(2, 3) = r[1];   T(2, 4) = -r[0];  T(2, 5) = 0.0;
    }
    
    return T;
}

template <int dim>
Tensor<1, dim> rotation_displacement(const Tensor<1, dim>& omega, const Tensor<1, dim>& r) {
    if constexpr (dim == 3) {
        return cross_product_3d(omega, r);
    } else {
        // 2D case: omega is scalar (rotation about z-axis)
        Tensor<1, dim> result;
        result[0] = -omega[0] * r[1];
        result[1] = omega[0] * r[0];
        return result;
    }
}

template <int dim>
Tensor<2, dim> skew_symmetric_matrix(const Tensor<1, dim>& r) {
    Tensor<2, dim> S;
    
    if constexpr (dim == 3) {
        S[0][0] = 0.0;    S[0][1] = -r[2];  S[0][2] = r[1];
        S[1][0] = r[2];   S[1][1] = 0.0;    S[1][2] = -r[0];
        S[2][0] = -r[1];  S[2][1] = r[0];   S[2][2] = 0.0;
    } else if constexpr (dim == 2) {
        S[0][0] = 0.0;    S[0][1] = -1.0;
        S[1][0] = 1.0;    S[1][1] = 0.0;
        // Scale by "r" magnitude for 2D (simplified)
    }
    
    return S;
}

// ============================================================================
// RBE2 Helpers
// ============================================================================

std::vector<std::vector<std::pair<unsigned int, double>>>
RBE2Parameters::compute_constraint_equations() const {
    std::vector<std::vector<std::pair<unsigned int, double>>> equations;
    
    // For each slave node
    for (size_t s = 0; s < slave_locations.size(); ++s) {
        // Compute offset from master
        Tensor<1, 3> r;
        for (unsigned int d = 0; d < 3; ++d) {
            r[d] = slave_locations[s][d] - master_location[d];
        }
        
        // For each constrained translational DOF
        for (unsigned int d = 0; d < 3; ++d) {
            if (!constrained_dofs[d]) continue;
            
            std::vector<std::pair<unsigned int, double>> eq;
            
            // Slave DOF coefficient = 1
            eq.emplace_back(s * 6 + d, 1.0);
            
            // Master translation coefficient = -1
            eq.emplace_back(slave_locations.size() * 6 + d, -1.0);
            
            // Master rotation contributions (if rotational DOFs exist)
            // u_x += wy*rz - wz*ry
            // u_y += wz*rx - wx*rz
            // u_z += wx*ry - wy*rx
            
            equations.push_back(eq);
        }
    }
    
    return equations;
}

RigidConnection create_rbe2_spider(
    const Point<3>& hub_center,
    const std::vector<Point<3>>& spoke_ends) {
    
    RigidConnection conn;
    conn.master_point = hub_center;
    conn.is_rigid = true;
    conn.description = "RBE2 spider (hub with " + 
                       std::to_string(spoke_ends.size()) + " spokes)";
    
    // Create boundary target from point list
    // Note: Actual implementation would need proper target type
    conn.slave_target = BoundaryTarget::from_point_list(spoke_ends);
    
    return conn;
}

RigidConnection create_rbe2_joint(
    const Point<3>& center,
    double radius,
    const BoundaryTarget& nodes) {
    
    RigidConnection conn;
    conn.master_point = center;
    conn.slave_target = nodes;
    conn.is_rigid = true;
    conn.node_search_tolerance = radius;
    conn.description = "RBE2 joint (r=" + std::to_string(radius) + ")";
    
    return conn;
}

// ============================================================================
// RBE3 Helpers
// ============================================================================

void RBE3Parameters::set_distance_weights() {
    weights.resize(node_locations.size());
    
    double total_inv_dist = 0.0;
    for (size_t i = 0; i < node_locations.size(); ++i) {
        double dist = reference_location.distance(node_locations[i]);
        double inv_dist = (dist > 1e-14) ? 1.0 / dist : 1e14;
        weights[i] = inv_dist;
        total_inv_dist += inv_dist;
    }
    
    // Normalize
    for (auto& w : weights) {
        w /= total_inv_dist;
    }
}

void RBE3Parameters::set_area_weights(const std::vector<double>& areas) {
    if (areas.size() != node_locations.size()) {
        throw std::runtime_error("Area weights size mismatch");
    }
    
    double total_area = std::accumulate(areas.begin(), areas.end(), 0.0);
    if (total_area < 1e-14) {
        set_uniform_weights();
        return;
    }
    
    weights.resize(areas.size());
    for (size_t i = 0; i < areas.size(); ++i) {
        weights[i] = areas[i] / total_area;
    }
}

RigidConnection create_rbe3_distribution(
    const Point<3>& reference,
    const BoundaryTarget& nodes,
    bool /*use_area_weights*/) {
    
    RigidConnection conn;
    conn.master_point = reference;
    conn.slave_target = nodes;
    conn.is_rigid = false;  // RBE3
    conn.description = "RBE3 load distribution";
    
    // Weights will be computed when applied based on actual node positions
    // If use_area_weights is true, caller should provide weights separately
    
    return conn;
}

RigidConnection create_rbe3_averaging(
    const Point<3>& reference,
    const BoundaryTarget& nodes) {
    
    RigidConnection conn;
    conn.master_point = reference;
    conn.slave_target = nodes;
    conn.is_rigid = false;
    conn.description = "RBE3 motion averaging";
    
    // Equal weights for averaging
    conn.slave_weights.clear();  // Will use equal weights when empty
    
    return conn;
}

// ============================================================================
// Inertia Properties
// ============================================================================

InertiaProperties InertiaProperties::compute(
    const std::vector<Point<3>>& points,
    const std::vector<double>& masses) {
    
    InertiaProperties props;
    
    if (points.empty()) {
        props.total_mass = 0.0;
        return props;
    }
    
    // Compute total mass and centroid
    props.total_mass = std::accumulate(masses.begin(), masses.end(), 0.0);
    
    if (props.total_mass < 1e-14) {
        return compute_uniform(points);
    }
    
    // Centroid = sum(m_i * r_i) / sum(m_i)
    for (size_t i = 0; i < points.size(); ++i) {
        for (unsigned int d = 0; d < 3; ++d) {
            props.centroid[d] += masses[i] * points[i][d];
        }
    }
    for (unsigned int d = 0; d < 3; ++d) {
        props.centroid[d] /= props.total_mass;
    }
    
    // Compute inertia tensor about centroid
    // I_ij = sum(m_k * (r_k^2 * delta_ij - r_ki * r_kj))
    for (size_t k = 0; k < points.size(); ++k) {
        Tensor<1, 3> r;
        for (unsigned int d = 0; d < 3; ++d) {
            r[d] = points[k][d] - props.centroid[d];
        }
        double r_sq = r * r;
        
        for (unsigned int i = 0; i < 3; ++i) {
            for (unsigned int j = 0; j < 3; ++j) {
                double delta_ij = (i == j) ? 1.0 : 0.0;
                props.inertia_tensor[i][j] += masses[k] * (r_sq * delta_ij - r[i] * r[j]);
            }
        }
    }
    
    // Note: Principal moments/directions computation would require
    // eigenvalue decomposition - placeholder for now
    props.principal_moments = Tensor<1, 3>({
        props.inertia_tensor[0][0],
        props.inertia_tensor[1][1],
        props.inertia_tensor[2][2]
    });
    props.principal_directions = unit_symmetric_tensor<3>();
    
    return props;
}

InertiaProperties InertiaProperties::compute_uniform(
    const std::vector<Point<3>>& points) {
    
    std::vector<double> unit_masses(points.size(), 1.0);
    return compute(points, unit_masses);
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template FullMatrix<double> rigid_body_transformation_matrix<3>(const Tensor<1, 3>& r);
template Tensor<1, 3> rotation_displacement<3>(const Tensor<1, 3>& omega, const Tensor<1, 3>& r);
template Tensor<2, 3> skew_symmetric_matrix<3>(const Tensor<1, 3>& r);

} // namespace FEA
