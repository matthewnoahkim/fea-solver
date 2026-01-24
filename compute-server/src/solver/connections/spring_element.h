#ifndef SPRING_ELEMENT_H
#define SPRING_ELEMENT_H

/**
 * @file spring_element.h
 * @brief Specialized spring element utilities
 * 
 * This header provides convenience functions and specialized implementations
 * for spring elements. The main spring connection types are defined in
 * constraint_base.h:
 * - SpringToGroundConnection
 * - SpringConnection  
 * - BushingConnection
 * 
 * This file adds:
 * - Spring stiffness calculation utilities
 * - Nonlinear spring support (future)
 * - Spring element analysis helpers
 */

#include "constraint_base.h"
#include <functional>

namespace FEA {

// ============================================================================
// Spring Stiffness Models
// ============================================================================

/**
 * @brief Linear spring force-displacement law
 */
struct LinearSpringLaw {
    double stiffness;
    
    double force(double displacement) const {
        return stiffness * displacement;
    }
    
    double tangent_stiffness(double /*displacement*/) const {
        return stiffness;
    }
};

/**
 * @brief Bilinear spring with tension/compression asymmetry
 */
struct BilinearSpringLaw {
    double tension_stiffness;
    double compression_stiffness;
    
    double force(double displacement) const {
        return (displacement >= 0) ? 
            tension_stiffness * displacement :
            compression_stiffness * displacement;
    }
    
    double tangent_stiffness(double displacement) const {
        return (displacement >= 0) ? tension_stiffness : compression_stiffness;
    }
};

/**
 * @brief Tension-only spring (no compression resistance)
 */
struct TensionOnlySpringLaw {
    double stiffness;
    
    double force(double displacement) const {
        return (displacement > 0) ? stiffness * displacement : 0.0;
    }
    
    double tangent_stiffness(double displacement) const {
        return (displacement > 0) ? stiffness : 0.0;
    }
};

/**
 * @brief Compression-only spring (no tension resistance)
 */
struct CompressionOnlySpringLaw {
    double stiffness;
    
    double force(double displacement) const {
        return (displacement < 0) ? stiffness * displacement : 0.0;
    }
    
    double tangent_stiffness(double displacement) const {
        return (displacement < 0) ? stiffness : 0.0;
    }
};

/**
 * @brief Spring with initial gap
 */
struct GapSpringLaw {
    double stiffness;
    double initial_gap;  // Positive = gap, negative = interference
    
    double force(double displacement) const {
        double effective_disp = displacement - initial_gap;
        return (effective_disp > 0) ? stiffness * effective_disp : 0.0;
    }
    
    double tangent_stiffness(double displacement) const {
        return (displacement > initial_gap) ? stiffness : 0.0;
    }
};

// ============================================================================
// Spring Element Utilities
// ============================================================================

/**
 * @brief Compute spring element stiffness matrix in local coordinates
 * 
 * Returns the 2x2 stiffness matrix for an axial spring:
 * K = k * [ 1  -1]
 *         [-1   1]
 */
inline FullMatrix<double> axial_spring_stiffness_local(double k) {
    FullMatrix<double> K(2, 2);
    K(0, 0) = k;
    K(0, 1) = -k;
    K(1, 0) = -k;
    K(1, 1) = k;
    return K;
}

/**
 * @brief Compute spring element stiffness matrix in global coordinates
 * 
 * @param k Spring stiffness
 * @param axis Unit vector along spring axis
 * @return 6x6 stiffness matrix for two 3D nodes
 */
template <int dim>
FullMatrix<double> axial_spring_stiffness_global(double k, const Tensor<1, dim>& axis) {
    FullMatrix<double> K(2 * dim, 2 * dim);
    
    // K = k * [n⊗n  -n⊗n; -n⊗n  n⊗n]
    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
            double k_ij = k * axis[i] * axis[j];
            
            // Node A - Node A
            K(i, j) = k_ij;
            // Node A - Node B
            K(i, dim + j) = -k_ij;
            // Node B - Node A
            K(dim + i, j) = -k_ij;
            // Node B - Node B
            K(dim + i, dim + j) = k_ij;
        }
    }
    
    return K;
}

/**
 * @brief Compute spring strain energy
 */
inline double spring_strain_energy(double stiffness, double elongation) {
    return 0.5 * stiffness * elongation * elongation;
}

/**
 * @brief Compute critical buckling load for compression spring
 * 
 * Uses Euler buckling formula: P_cr = pi^2 * EI / L^2
 * For a spring with effective bending stiffness EI
 */
inline double spring_buckling_load(double bending_stiffness, double length) {
    const double pi = 3.14159265358979323846;
    return pi * pi * bending_stiffness / (length * length);
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create spring to ground with nonlinear stiffness (placeholder)
 * 
 * @note Nonlinear springs require iterative solution - see NewtonRaphsonSolver
 */
SpringToGroundConnection create_nonlinear_spring_to_ground(
    const BoundaryTarget& target,
    std::function<double(double)> force_law,
    const std::string& description = "Nonlinear spring to ground");

/**
 * @brief Create series of springs between points
 */
std::vector<SpringConnection> create_spring_chain(
    const std::vector<Point<3>>& points,
    double stiffness);

/**
 * @brief Create parallel spring arrangement
 * 
 * Multiple springs between same two points with combined stiffness
 */
SpringConnection create_parallel_springs(
    const Point<3>& a, const Point<3>& b,
    const std::vector<double>& stiffnesses);

} // namespace FEA

#endif // SPRING_ELEMENT_H
