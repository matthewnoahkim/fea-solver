/**
 * @file newton_raphson.h
 * @brief Newton-Raphson nonlinear solver
 */

#ifndef FEA_NEWTON_RAPHSON_H
#define FEA_NEWTON_RAPHSON_H

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <functional>
#include <string>

namespace FEA {

/**
 * @brief Newton-Raphson solver configuration
 */
struct NewtonRaphsonConfig {
    double tolerance = 1e-8;          // Residual norm tolerance
    double relative_tolerance = 1e-6; // Relative residual tolerance
    unsigned int max_iterations = 20;
    
    bool use_line_search = true;
    double line_search_alpha_min = 0.1;
    double line_search_c = 1e-4;      // Armijo constant
    
    bool adaptive_step = false;
    double initial_step_size = 1.0;
    
    // Output
    bool verbose = true;
};

/**
 * @brief Newton-Raphson convergence status
 */
struct NewtonRaphsonStatus {
    bool converged = false;
    unsigned int iterations = 0;
    double final_residual = 0;
    double initial_residual = 0;
    std::string message;
};

/**
 * @brief Newton-Raphson nonlinear solver
 * 
 * Solves R(u) = 0 using Newton-Raphson iteration:
 * K(u_n) Δu = -R(u_n)
 * u_{n+1} = u_n + α Δu
 */
class NewtonRaphsonSolver {
public:
    using ResidualFunction = std::function<void(
        const dealii::Vector<double> &solution,
        dealii::Vector<double> &residual)>;
    
    using TangentFunction = std::function<void(
        const dealii::Vector<double> &solution,
        dealii::SparseMatrix<double> &tangent)>;
    
    using LinearSolverFunction = std::function<void(
        const dealii::SparseMatrix<double> &matrix,
        const dealii::Vector<double> &rhs,
        dealii::Vector<double> &solution)>;
    
    /**
     * @brief Constructor
     * @param config Solver configuration
     */
    explicit NewtonRaphsonSolver(const NewtonRaphsonConfig &config = NewtonRaphsonConfig());
    
    /**
     * @brief Set residual computation function
     */
    void set_residual_function(ResidualFunction func) { compute_residual_ = func; }
    
    /**
     * @brief Set tangent matrix computation function
     */
    void set_tangent_function(TangentFunction func) { compute_tangent_ = func; }
    
    /**
     * @brief Set linear solver function
     */
    void set_linear_solver(LinearSolverFunction func) { solve_linear_ = func; }
    
    /**
     * @brief Set constraint matrix
     */
    void set_constraints(const dealii::AffineConstraints<double> *constraints) {
        constraints_ = constraints;
    }
    
    /**
     * @brief Solve the nonlinear system
     * @param solution Initial guess and final solution
     * @param tangent_matrix Tangent stiffness matrix (will be modified)
     * @return Convergence status
     */
    NewtonRaphsonStatus solve(dealii::Vector<double> &solution,
                               dealii::SparseMatrix<double> &tangent_matrix);
    
    /**
     * @brief Get iteration history
     */
    const std::vector<double>& get_residual_history() const { return residual_history_; }

private:
    double compute_residual_norm(const dealii::Vector<double> &residual) const;
    double line_search(dealii::Vector<double> &solution,
                       const dealii::Vector<double> &delta_u,
                       const dealii::Vector<double> &residual,
                       dealii::Vector<double> &new_residual);
    
    NewtonRaphsonConfig config_;
    
    ResidualFunction compute_residual_;
    TangentFunction compute_tangent_;
    LinearSolverFunction solve_linear_;
    
    const dealii::AffineConstraints<double> *constraints_ = nullptr;
    
    std::vector<double> residual_history_;
};

} // namespace FEA

#endif // FEA_NEWTON_RAPHSON_H
