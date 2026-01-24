/**
 * @file newton_raphson.cc
 * @brief Implementation of Newton-Raphson solver
 */

#include "newton_raphson.h"
#include <iostream>
#include <cmath>

namespace FEA {

NewtonRaphsonSolver::NewtonRaphsonSolver(const NewtonRaphsonConfig &config)
    : config_(config)
{
}

double NewtonRaphsonSolver::compute_residual_norm(
    const dealii::Vector<double> &residual) const 
{
    return residual.l2_norm();
}

double NewtonRaphsonSolver::line_search(
    dealii::Vector<double> &solution,
    const dealii::Vector<double> &delta_u,
    const dealii::Vector<double> &residual,
    dealii::Vector<double> &new_residual) 
{
    double alpha = 1.0;
    double res0 = compute_residual_norm(residual);
    
    dealii::Vector<double> trial_solution(solution.size());
    
    while (alpha > config_.line_search_alpha_min) {
        // Trial solution
        trial_solution = solution;
        trial_solution.add(alpha, delta_u);
        
        if (constraints_) {
            constraints_->distribute(trial_solution);
        }
        
        // Compute residual at trial
        compute_residual_(trial_solution, new_residual);
        
        double res_trial = compute_residual_norm(new_residual);
        
        // Armijo condition
        if (res_trial <= res0 * (1.0 - config_.line_search_c * alpha)) {
            solution = trial_solution;
            return alpha;
        }
        
        alpha *= 0.5;
    }
    
    // Accept whatever we got
    solution = trial_solution;
    return alpha;
}

NewtonRaphsonStatus NewtonRaphsonSolver::solve(
    dealii::Vector<double> &solution,
    dealii::SparseMatrix<double> &tangent_matrix) 
{
    NewtonRaphsonStatus status;
    residual_history_.clear();
    
    if (!compute_residual_ || !compute_tangent_ || !solve_linear_) {
        status.message = "Solver functions not set";
        return status;
    }
    
    const unsigned int n_dofs = solution.size();
    dealii::Vector<double> residual(n_dofs);
    dealii::Vector<double> delta_u(n_dofs);
    dealii::Vector<double> new_residual(n_dofs);
    
    // Initial residual
    compute_residual_(solution, residual);
    status.initial_residual = compute_residual_norm(residual);
    residual_history_.push_back(status.initial_residual);
    
    if (config_.verbose) {
        std::cout << "Newton-Raphson iteration 0: residual = " 
                  << status.initial_residual << std::endl;
    }
    
    // Check initial convergence
    if (status.initial_residual < config_.tolerance) {
        status.converged = true;
        status.final_residual = status.initial_residual;
        status.message = "Converged (initial)";
        return status;
    }
    
    // Newton iterations
    for (unsigned int iter = 0; iter < config_.max_iterations; ++iter) {
        status.iterations = iter + 1;
        
        // Compute tangent matrix
        compute_tangent_(solution, tangent_matrix);
        
        // Apply constraints to residual for linear solve
        dealii::Vector<double> constrained_rhs = residual;
        constrained_rhs *= -1.0;
        
        // Solve linear system: K Δu = -R
        delta_u = 0;
        solve_linear_(tangent_matrix, constrained_rhs, delta_u);
        
        if (constraints_) {
            constraints_->distribute(delta_u);
        }
        
        // Line search
        double alpha = 1.0;
        if (config_.use_line_search) {
            alpha = line_search(solution, delta_u, residual, new_residual);
            residual = new_residual;
        } else {
            solution.add(alpha, delta_u);
            if (constraints_) {
                constraints_->distribute(solution);
            }
            compute_residual_(solution, residual);
        }
        
        double res_norm = compute_residual_norm(residual);
        residual_history_.push_back(res_norm);
        
        if (config_.verbose) {
            std::cout << "Newton-Raphson iteration " << iter + 1 
                      << ": residual = " << res_norm
                      << ", alpha = " << alpha << std::endl;
        }
        
        // Check convergence
        bool abs_conv = res_norm < config_.tolerance;
        bool rel_conv = res_norm < config_.relative_tolerance * status.initial_residual;
        
        if (abs_conv || rel_conv) {
            status.converged = true;
            status.final_residual = res_norm;
            status.message = abs_conv ? "Converged (absolute)" : "Converged (relative)";
            return status;
        }
        
        // Check for divergence
        if (std::isnan(res_norm) || std::isinf(res_norm)) {
            status.final_residual = res_norm;
            status.message = "Diverged (NaN/Inf)";
            return status;
        }
        
        if (res_norm > 1e6 * status.initial_residual) {
            status.final_residual = res_norm;
            status.message = "Diverged (residual growth)";
            return status;
        }
    }
    
    status.final_residual = compute_residual_norm(residual);
    status.message = "Max iterations reached";
    return status;
}

} // namespace FEA
