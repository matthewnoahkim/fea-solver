/**
 * @file line_search.cc
 * @brief Implementation of line search algorithms
 */

#include "line_search.h"
#include <cmath>
#include <algorithm>

namespace FEA {

LineSearch::LineSearch(const LineSearchConfig &config)
    : config_(config)
{
}

LineSearchResult LineSearch::backtracking(
    const dealii::Vector<double> &x,
    const dealii::Vector<double> &direction,
    MeritFunction merit,
    double merit_0) 
{
    LineSearchResult result;
    result.alpha = config_.alpha_max;
    
    dealii::Vector<double> x_trial(x.size());
    
    for (unsigned int iter = 0; iter < config_.max_iterations; ++iter) {
        result.iterations = iter + 1;
        
        // Compute trial point
        x_trial = x;
        x_trial.add(result.alpha, direction);
        
        // Evaluate merit function
        result.merit_value = merit(x_trial);
        
        // Check Armijo condition
        if (result.merit_value <= merit_0 * (1.0 - config_.c1 * result.alpha)) {
            result.success = true;
            return result;
        }
        
        // Reduce step size
        result.alpha *= 0.5;
        
        if (result.alpha < config_.alpha_min) {
            result.alpha = config_.alpha_min;
            x_trial = x;
            x_trial.add(result.alpha, direction);
            result.merit_value = merit(x_trial);
            result.success = result.merit_value < merit_0;
            return result;
        }
    }
    
    return result;
}

LineSearchResult LineSearch::cubic_interpolation(
    const dealii::Vector<double> &x,
    const dealii::Vector<double> &direction,
    MeritFunction merit,
    double merit_0,
    double grad_dot_dir) 
{
    LineSearchResult result;
    
    double alpha0 = 0;
    double alpha1 = config_.alpha_max;
    double f0 = merit_0;
    double fp0 = grad_dot_dir;  // Directional derivative at alpha=0
    
    dealii::Vector<double> x_trial(x.size());
    
    // First trial
    x_trial = x;
    x_trial.add(alpha1, direction);
    double f1 = merit(x_trial);
    result.iterations = 1;
    
    // Check if first step is acceptable
    if (f1 <= f0 + config_.c1 * alpha1 * fp0) {
        result.alpha = alpha1;
        result.merit_value = f1;
        result.success = true;
        return result;
    }
    
    // Quadratic interpolation for first iteration
    double alpha2 = -fp0 * alpha1 * alpha1 / (2.0 * (f1 - f0 - fp0 * alpha1));
    alpha2 = std::max(alpha2, 0.1 * alpha1);
    alpha2 = std::min(alpha2, 0.9 * alpha1);
    
    for (unsigned int iter = 1; iter < config_.max_iterations; ++iter) {
        result.iterations = iter + 1;
        
        x_trial = x;
        x_trial.add(alpha2, direction);
        double f2 = merit(x_trial);
        
        // Check Armijo condition
        if (f2 <= f0 + config_.c1 * alpha2 * fp0) {
            result.alpha = alpha2;
            result.merit_value = f2;
            result.success = true;
            return result;
        }
        
        // Cubic interpolation
        double d1 = f1 - f0 - fp0 * alpha1;
        double d2 = f2 - f0 - fp0 * alpha2;
        
        double denom = alpha1 * alpha1 * alpha2 * alpha2 * (alpha2 - alpha1);
        if (std::abs(denom) < 1e-20) break;
        
        double a = (alpha1 * alpha1 * d2 - alpha2 * alpha2 * d1) / denom;
        double b = (-alpha1 * alpha1 * alpha1 * d2 + alpha2 * alpha2 * alpha2 * d1) / denom;
        
        double discriminant = b * b - 3.0 * a * fp0;
        double alpha_new;
        
        if (std::abs(a) < 1e-20) {
            // Quadratic case
            alpha_new = -fp0 / (2.0 * b);
        } else if (discriminant < 0) {
            alpha_new = 0.5 * alpha2;
        } else {
            alpha_new = (-b + std::sqrt(discriminant)) / (3.0 * a);
        }
        
        // Safeguard
        alpha_new = std::max(alpha_new, 0.1 * alpha2);
        alpha_new = std::min(alpha_new, 0.5 * alpha2);
        
        if (alpha_new < config_.alpha_min) {
            result.alpha = config_.alpha_min;
            x_trial = x;
            x_trial.add(result.alpha, direction);
            result.merit_value = merit(x_trial);
            result.success = result.merit_value < f0;
            return result;
        }
        
        alpha1 = alpha2;
        f1 = f2;
        alpha2 = alpha_new;
    }
    
    // Return best found
    result.alpha = alpha2;
    x_trial = x;
    x_trial.add(result.alpha, direction);
    result.merit_value = merit(x_trial);
    result.success = result.merit_value < merit_0;
    return result;
}

LineSearchResult LineSearch::wolfe(
    const dealii::Vector<double> &x,
    const dealii::Vector<double> &direction,
    MeritFunction merit,
    GradientFunction gradient,
    double merit_0,
    double grad_dot_dir) 
{
    LineSearchResult result;
    
    double alpha_lo = 0;
    double alpha_hi = config_.alpha_max;
    double f_lo = merit_0;
    double fp_lo = grad_dot_dir;
    
    dealii::Vector<double> x_trial(x.size());
    dealii::Vector<double> grad_trial(x.size());
    
    double alpha = config_.alpha_max;
    
    for (unsigned int iter = 0; iter < config_.max_iterations; ++iter) {
        result.iterations = iter + 1;
        
        x_trial = x;
        x_trial.add(alpha, direction);
        double f_alpha = merit(x_trial);
        
        // Check Armijo (sufficient decrease)
        if (f_alpha > merit_0 + config_.c1 * alpha * grad_dot_dir ||
            (iter > 0 && f_alpha >= f_lo)) {
            alpha_hi = alpha;
            alpha = 0.5 * (alpha_lo + alpha_hi);
            continue;
        }
        
        // Compute gradient at trial point
        gradient(x_trial, grad_trial);
        double fp_alpha = grad_trial * direction;
        
        // Check curvature condition
        if (std::abs(fp_alpha) <= -config_.c2 * grad_dot_dir) {
            result.alpha = alpha;
            result.merit_value = f_alpha;
            result.success = true;
            return result;
        }
        
        if (fp_alpha >= 0) {
            alpha_hi = alpha;
        } else {
            alpha_lo = alpha;
            f_lo = f_alpha;
            fp_lo = fp_alpha;
        }
        
        alpha = 0.5 * (alpha_lo + alpha_hi);
        
        if (alpha_hi - alpha_lo < config_.alpha_min) {
            break;
        }
    }
    
    result.alpha = alpha;
    x_trial = x;
    x_trial.add(result.alpha, direction);
    result.merit_value = merit(x_trial);
    result.success = result.merit_value < merit_0;
    return result;
}

} // namespace FEA
