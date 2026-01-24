/**
 * @file line_search.h
 * @brief Line search algorithms for nonlinear solvers
 */

#ifndef FEA_LINE_SEARCH_H
#define FEA_LINE_SEARCH_H

#include <deal.II/lac/vector.h>
#include <functional>

namespace FEA {

/**
 * @brief Line search configuration
 */
struct LineSearchConfig {
    double alpha_min = 0.01;      // Minimum step size
    double alpha_max = 1.0;       // Maximum step size
    double c1 = 1e-4;            // Armijo constant (sufficient decrease)
    double c2 = 0.9;             // Curvature condition constant
    unsigned int max_iterations = 20;
    bool use_wolfe = false;      // Use strong Wolfe conditions
};

/**
 * @brief Line search result
 */
struct LineSearchResult {
    double alpha = 0;           // Step size found
    double merit_value = 0;     // Merit function value at new point
    unsigned int iterations = 0;
    bool success = false;
};

/**
 * @brief Line search algorithms
 */
class LineSearch {
public:
    using MeritFunction = std::function<double(const dealii::Vector<double> &)>;
    using GradientFunction = std::function<void(const dealii::Vector<double> &,
                                                 dealii::Vector<double> &)>;
    
    explicit LineSearch(const LineSearchConfig &config = LineSearchConfig());
    
    /**
     * @brief Perform backtracking line search
     * @param x Current point
     * @param direction Search direction
     * @param merit Merit function (typically residual norm)
     * @param merit_0 Merit value at current point
     * @return Line search result
     */
    LineSearchResult backtracking(
        const dealii::Vector<double> &x,
        const dealii::Vector<double> &direction,
        MeritFunction merit,
        double merit_0);
    
    /**
     * @brief Perform cubic interpolation line search
     */
    LineSearchResult cubic_interpolation(
        const dealii::Vector<double> &x,
        const dealii::Vector<double> &direction,
        MeritFunction merit,
        double merit_0,
        double grad_dot_dir);
    
    /**
     * @brief Perform line search with Wolfe conditions
     */
    LineSearchResult wolfe(
        const dealii::Vector<double> &x,
        const dealii::Vector<double> &direction,
        MeritFunction merit,
        GradientFunction gradient,
        double merit_0,
        double grad_dot_dir);

private:
    LineSearchConfig config_;
};

} // namespace FEA

#endif // FEA_LINE_SEARCH_H
