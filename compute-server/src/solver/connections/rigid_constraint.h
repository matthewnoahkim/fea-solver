#ifndef RIGID_CONSTRAINT_H
#define RIGID_CONSTRAINT_H

/**
 * @file rigid_constraint.h
 * @brief Rigid body constraint utilities (RBE2/RBE3)
 * 
 * This header provides utilities for rigid body constraints.
 * The main rigid connection type is defined in constraint_base.h:
 * - RigidConnection (for both RBE2 and RBE3)
 * 
 * This file adds:
 * - Rigid body motion calculation helpers
 * - Constraint equation generation utilities
 * - Rigid region detection
 */

#include "constraint_base.h"
#include <deal.II/base/symmetric_tensor.h>

namespace FEA {

// ============================================================================
// Rigid Body Motion Utilities
// ============================================================================

/**
 * @brief Compute rigid body transformation matrix
 * 
 * For a point at position r relative to reference point,
 * computes the 3x6 transformation matrix T such that:
 * u = T * [u_ref; omega]
 * 
 * where u_ref is the translation and omega is the rotation vector
 */
template <int dim>
FullMatrix<double> rigid_body_transformation_matrix(const Tensor<1, dim>& r);

/**
 * @brief Compute rotation contribution to displacement
 * 
 * u_rotation = omega × r
 */
template <int dim>
Tensor<1, dim> rotation_displacement(const Tensor<1, dim>& omega, const Tensor<1, dim>& r);

/**
 * @brief Compute the skew-symmetric matrix [r]× for cross product
 * 
 * [r]× = [ 0   -rz   ry ]
 *        [ rz   0   -rx ]
 *        [-ry   rx   0  ]
 * 
 * Such that [r]× * omega = r × omega
 */
template <int dim>
Tensor<2, dim> skew_symmetric_matrix(const Tensor<1, dim>& r);

// ============================================================================
// RBE2 (Rigid Link) Helpers
// ============================================================================

/**
 * @brief Parameters for RBE2 constraint
 */
struct RBE2Parameters {
    Point<3> master_location;
    std::vector<Point<3>> slave_locations;
    std::array<bool, 6> constrained_dofs = {true, true, true, true, true, true};
    double tolerance = 1e-6;
    
    /**
     * @brief Compute constraint equations for RBE2
     * 
     * For each slave node s and each DOF d:
     * u_slave[d] = u_master[d] + contribution from rotation
     * 
     * @return Vector of constraint equation coefficients
     */
    std::vector<std::vector<std::pair<unsigned int, double>>> 
    compute_constraint_equations() const;
};

/**
 * @brief Create RBE2 from spider geometry
 * 
 * Creates a rigid connection from a central hub to multiple spoke endpoints
 */
RigidConnection create_rbe2_spider(
    const Point<3>& hub_center,
    const std::vector<Point<3>>& spoke_ends);

/**
 * @brief Create RBE2 for rigid joint
 * 
 * Connects all nodes within a spherical region to a master node
 */
RigidConnection create_rbe2_joint(
    const Point<3>& center,
    double radius,
    const BoundaryTarget& nodes);

// ============================================================================
// RBE3 (Distributing Link) Helpers
// ============================================================================

/**
 * @brief Parameters for RBE3 constraint
 */
struct RBE3Parameters {
    Point<3> reference_location;
    std::vector<Point<3>> node_locations;
    std::vector<double> weights;
    std::array<bool, 6> constrained_dofs = {true, true, true, true, true, true};
    double tolerance = 1e-6;
    
    /**
     * @brief Compute weight factors for uniform distribution
     */
    void set_uniform_weights() {
        weights.assign(node_locations.size(), 1.0 / node_locations.size());
    }
    
    /**
     * @brief Compute weight factors based on distance from reference
     */
    void set_distance_weights();
    
    /**
     * @brief Compute weight factors based on tributary area
     */
    void set_area_weights(const std::vector<double>& areas);
};

/**
 * @brief Create RBE3 for load distribution
 * 
 * Distributes motion from multiple nodes to a reference point
 */
RigidConnection create_rbe3_distribution(
    const Point<3>& reference,
    const BoundaryTarget& nodes,
    bool use_area_weights = false);

/**
 * @brief Create RBE3 for averaging motion
 * 
 * Reference node follows average motion of connected nodes
 */
RigidConnection create_rbe3_averaging(
    const Point<3>& reference,
    const BoundaryTarget& nodes);

// ============================================================================
// Rigid Region Utilities
// ============================================================================

/**
 * @brief Detect potential rigid regions in mesh
 * 
 * Identifies groups of nodes that should be rigidly connected
 * based on geometric proximity or material properties
 */
struct RigidRegionDetector {
    double proximity_tolerance = 1e-3;
    double stiffness_ratio_threshold = 1000.0;
    
    /**
     * @brief Find nodes that form a near-rigid cluster
     */
    template <int dim>
    std::vector<std::vector<Point<dim>>> detect_rigid_clusters(
        const DoFHandler<dim>& dof_handler) const;
    
    /**
     * @brief Suggest RBE2 connections for detected rigid regions
     */
    template <int dim>
    std::vector<RigidConnection> suggest_rigid_connections(
        const DoFHandler<dim>& dof_handler) const;
};

/**
 * @brief Compute inertia properties of a set of points
 * 
 * Used for determining appropriate master node location
 */
struct InertiaProperties {
    Point<3> centroid;
    double total_mass;
    Tensor<2, 3> inertia_tensor;
    Tensor<1, 3> principal_moments;
    Tensor<2, 3> principal_directions;
    
    static InertiaProperties compute(
        const std::vector<Point<3>>& points,
        const std::vector<double>& masses);
    
    static InertiaProperties compute_uniform(
        const std::vector<Point<3>>& points);
};

} // namespace FEA

#endif // RIGID_CONSTRAINT_H
