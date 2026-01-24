/**
 * @file tensor_utils.h
 * @brief Tensor manipulation utilities
 */

#ifndef FEA_TENSOR_UTILS_H
#define FEA_TENSOR_UTILS_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <array>

namespace FEA {

/**
 * @brief Tensor utilities
 */
class TensorUtils {
public:
    static constexpr unsigned int dim = 3;
    
    /**
     * @brief Convert symmetric tensor to Voigt notation vector
     */
    static std::array<double, 6> to_voigt(const dealii::SymmetricTensor<2, dim> &tensor);
    
    /**
     * @brief Convert Voigt vector to symmetric tensor
     */
    static dealii::SymmetricTensor<2, dim> from_voigt(const std::array<double, 6> &voigt);
    
    /**
     * @brief Compute rotation matrix from axis-angle
     */
    static dealii::Tensor<2, dim> rotation_matrix(
        const dealii::Tensor<1, dim> &axis, double angle);
    
    /**
     * @brief Compute rotation matrix from Euler angles (ZXZ convention)
     */
    static dealii::Tensor<2, dim> euler_rotation(double phi, double theta, double psi);
    
    /**
     * @brief Rotate a symmetric tensor
     */
    static dealii::SymmetricTensor<2, dim> rotate_tensor(
        const dealii::SymmetricTensor<2, dim> &tensor,
        const dealii::Tensor<2, dim> &rotation);
    
    /**
     * @brief Rotate a fourth-order tensor
     */
    static dealii::SymmetricTensor<4, dim> rotate_tensor(
        const dealii::SymmetricTensor<4, dim> &tensor,
        const dealii::Tensor<2, dim> &rotation);
    
    /**
     * @brief Compute deviator of a tensor
     */
    static dealii::SymmetricTensor<2, dim> deviator(
        const dealii::SymmetricTensor<2, dim> &tensor);
    
    /**
     * @brief Compute effective (von Mises) value
     */
    static double effective_value(const dealii::SymmetricTensor<2, dim> &tensor);
};

} // namespace FEA

#endif // FEA_TENSOR_UTILS_H
