#ifndef REACTION_FORCES_H
#define REACTION_FORCES_H

/**
 * @file reaction_forces.h
 * @brief Reaction force computation and equilibrium verification
 * 
 * Computes reaction forces at constrained degrees of freedom and
 * verifies global force and moment equilibrium.
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_system.h>

#include <nlohmann/json.hpp>
#include <map>
#include <vector>
#include <set>
#include <string>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

/**
 * @brief Computes reaction forces and checks equilibrium
 * 
 * @tparam dim Spatial dimension (2 or 3)
 */
template <int dim>
class ReactionForceCalculator {
public:
    /**
     * @brief Reaction force summary for a boundary region
     */
    struct BoundaryReaction {
        unsigned int boundary_id;
        std::string description;
        Tensor<1, dim> force;           ///< Total reaction force [N]
        Tensor<1, dim> moment;          ///< Total moment about centroid [N·m]
        Point<dim> centroid;            ///< Geometric centroid of constrained nodes
        unsigned int num_nodes;         ///< Number of constrained nodes
        double area;                    ///< Approximate boundary area
        
        json to_json() const {
            json j;
            j["boundary_id"] = boundary_id;
            j["description"] = description;
            j["force"] = std::vector<double>{force[0], force[1], 
                                              dim == 3 ? force[2] : 0.0};
            j["moment"] = std::vector<double>{moment[0], moment[1],
                                               dim == 3 ? moment[2] : 0.0};
            j["centroid"] = std::vector<double>{centroid[0], centroid[1],
                                                 dim == 3 ? centroid[2] : 0.0};
            j["num_nodes"] = num_nodes;
            j["area"] = area;
            return j;
        }
    };
    
    /**
     * @brief Reaction force at a single node
     */
    struct NodalReaction {
        unsigned int node_id;
        Point<dim> location;
        Tensor<1, dim> force;
        std::array<bool, dim> constrained_dofs;
        
        json to_json() const {
            json j;
            j["node_id"] = node_id;
            j["location"] = std::vector<double>{location[0], location[1],
                                                 dim == 3 ? location[2] : 0.0};
            j["force"] = std::vector<double>{force[0], force[1],
                                              dim == 3 ? force[2] : 0.0};
            return j;
        }
    };
    
    /**
     * @brief Equilibrium verification results
     */
    struct EquilibriumCheck {
        Tensor<1, dim> total_applied_force;     ///< Sum of external loads
        Tensor<1, dim> total_reaction_force;    ///< Sum of reactions
        Tensor<1, dim> total_applied_moment;    ///< Applied moment about origin
        Tensor<1, dim> total_reaction_moment;   ///< Reaction moment about origin
        Tensor<1, dim> force_residual;          ///< Force imbalance
        Tensor<1, dim> moment_residual;         ///< Moment imbalance
        double force_error_percent;             ///< % error in force balance
        double moment_error_percent;            ///< % error in moment balance
        bool is_balanced;                       ///< True if residuals are acceptable
        
        json to_json() const {
            return {
                {"total_applied_force", std::vector<double>{total_applied_force[0],
                    total_applied_force[1], dim == 3 ? total_applied_force[2] : 0.0}},
                {"total_reaction_force", std::vector<double>{total_reaction_force[0],
                    total_reaction_force[1], dim == 3 ? total_reaction_force[2] : 0.0}},
                {"force_residual", std::vector<double>{force_residual[0],
                    force_residual[1], dim == 3 ? force_residual[2] : 0.0}},
                {"force_error_percent", force_error_percent},
                {"moment_error_percent", moment_error_percent},
                {"is_balanced", is_balanced}
            };
        }
    };
    
    /**
     * @brief Construct calculator
     */
    ReactionForceCalculator(const DoFHandler<dim>& dof_handler,
                            const Mapping<dim>& mapping,
                            const AffineConstraints<double>& constraints);
    
    /**
     * @brief Compute reactions from R = K*u - F at constrained DOFs
     * @param K System stiffness matrix
     * @param u Displacement solution
     * @param F External force vector (assembled RHS)
     */
    void compute(const SparseMatrix<double>& K,
                 const Vector<double>& u,
                 const Vector<double>& F);
    
    // =========================================================================
    // Result Accessors
    // =========================================================================
    
    /**
     * @brief Get reactions grouped by boundary
     */
    std::vector<BoundaryReaction> get_boundary_reactions() const;
    
    /**
     * @brief Get reactions at individual nodes
     */
    std::vector<NodalReaction> get_nodal_reactions() const;
    
    /**
     * @brief Get total reaction force
     */
    Tensor<1, dim> get_total_force() const { return total_force_; }
    
    /**
     * @brief Get total reaction moment about a point
     */
    Tensor<1, dim> get_total_moment(const Point<dim>& about = Point<dim>()) const;
    
    /**
     * @brief Check force/moment equilibrium
     */
    EquilibriumCheck check_equilibrium(
        const Vector<double>& F,
        const Point<dim>& moment_center = Point<dim>()) const;
    
    /**
     * @brief Get reaction at specific boundary
     */
    BoundaryReaction get_reaction_at_boundary(unsigned int boundary_id) const;
    
    /**
     * @brief Get raw reaction vector (all DOFs)
     */
    const Vector<double>& get_reaction_vector() const { return reaction_vector_; }
    
    /**
     * @brief Get results as JSON
     */
    json to_json() const;
    
private:
    const DoFHandler<dim>& dof_handler_;
    const Mapping<dim>& mapping_;
    const AffineConstraints<double>& constraints_;
    
    Vector<double> reaction_vector_;
    std::map<unsigned int, BoundaryReaction> boundary_reactions_;
    std::vector<NodalReaction> nodal_reactions_;
    Tensor<1, dim> total_force_;
    Tensor<1, dim> total_moment_;
    
    /**
     * @brief Group nodal reactions by boundary ID
     */
    void categorize_by_boundary();
    
    /**
     * @brief Compute moments about boundary centroids
     */
    void compute_boundary_moments();
};

} // namespace FEA

#endif // REACTION_FORCES_H
