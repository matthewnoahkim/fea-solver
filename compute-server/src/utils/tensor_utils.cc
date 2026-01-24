/**
 * @file tensor_utils.cc
 * @brief Implementation of tensor utilities
 */

#include "tensor_utils.h"
#include <cmath>

namespace FEA {

std::array<double, 6> TensorUtils::to_voigt(const dealii::SymmetricTensor<2, dim> &tensor) {
    // Voigt notation: [σ11, σ22, σ33, σ23, σ13, σ12]
    return {
        tensor[0][0], tensor[1][1], tensor[2][2],
        tensor[1][2], tensor[0][2], tensor[0][1]
    };
}

dealii::SymmetricTensor<2, dim> TensorUtils::from_voigt(const std::array<double, 6> &voigt) {
    dealii::SymmetricTensor<2, dim> tensor;
    tensor[0][0] = voigt[0];
    tensor[1][1] = voigt[1];
    tensor[2][2] = voigt[2];
    tensor[1][2] = voigt[3];
    tensor[0][2] = voigt[4];
    tensor[0][1] = voigt[5];
    return tensor;
}

dealii::Tensor<2, dim> TensorUtils::rotation_matrix(
    const dealii::Tensor<1, dim> &axis, double angle) 
{
    dealii::Tensor<1, dim> n = axis / axis.norm();
    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;
    
    dealii::Tensor<2, dim> R;
    R[0][0] = t * n[0] * n[0] + c;
    R[0][1] = t * n[0] * n[1] - s * n[2];
    R[0][2] = t * n[0] * n[2] + s * n[1];
    R[1][0] = t * n[0] * n[1] + s * n[2];
    R[1][1] = t * n[1] * n[1] + c;
    R[1][2] = t * n[1] * n[2] - s * n[0];
    R[2][0] = t * n[0] * n[2] - s * n[1];
    R[2][1] = t * n[1] * n[2] + s * n[0];
    R[2][2] = t * n[2] * n[2] + c;
    
    return R;
}

dealii::Tensor<2, dim> TensorUtils::euler_rotation(double phi, double theta, double psi) {
    double c1 = std::cos(phi), s1 = std::sin(phi);
    double c2 = std::cos(theta), s2 = std::sin(theta);
    double c3 = std::cos(psi), s3 = std::sin(psi);
    
    dealii::Tensor<2, dim> R;
    R[0][0] = c1*c3 - c2*s1*s3;
    R[0][1] = -c1*s3 - c2*c3*s1;
    R[0][2] = s1*s2;
    R[1][0] = c3*s1 + c1*c2*s3;
    R[1][1] = c1*c2*c3 - s1*s3;
    R[1][2] = -c1*s2;
    R[2][0] = s2*s3;
    R[2][1] = c3*s2;
    R[2][2] = c2;
    
    return R;
}

dealii::SymmetricTensor<2, dim> TensorUtils::rotate_tensor(
    const dealii::SymmetricTensor<2, dim> &tensor,
    const dealii::Tensor<2, dim> &rotation) 
{
    // T' = R T R^T
    dealii::SymmetricTensor<2, dim> result;
    
    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = i; j < dim; ++j) {
            double val = 0;
            for (unsigned int k = 0; k < dim; ++k) {
                for (unsigned int l = 0; l < dim; ++l) {
                    val += rotation[i][k] * tensor[k][l] * rotation[j][l];
                }
            }
            result[i][j] = val;
        }
    }
    
    return result;
}

dealii::SymmetricTensor<4, dim> TensorUtils::rotate_tensor(
    const dealii::SymmetricTensor<4, dim> &tensor,
    const dealii::Tensor<2, dim> &rotation) 
{
    // C'_ijkl = R_im R_jn R_ko R_lp C_mnop
    dealii::SymmetricTensor<4, dim> result;
    
    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = i; j < dim; ++j) {
            for (unsigned int k = 0; k < dim; ++k) {
                for (unsigned int l = k; l < dim; ++l) {
                    double val = 0;
                    for (unsigned int m = 0; m < dim; ++m) {
                        for (unsigned int n = 0; n < dim; ++n) {
                            for (unsigned int o = 0; o < dim; ++o) {
                                for (unsigned int p = 0; p < dim; ++p) {
                                    val += rotation[i][m] * rotation[j][n] *
                                           rotation[k][o] * rotation[l][p] *
                                           tensor[m][n][o][p];
                                }
                            }
                        }
                    }
                    result[i][j][k][l] = val;
                }
            }
        }
    }
    
    return result;
}

dealii::SymmetricTensor<2, dim> TensorUtils::deviator(
    const dealii::SymmetricTensor<2, dim> &tensor) 
{
    double tr = dealii::trace(tensor) / 3.0;
    dealii::SymmetricTensor<2, dim> dev = tensor;
    for (unsigned int i = 0; i < dim; ++i) {
        dev[i][i] -= tr;
    }
    return dev;
}

double TensorUtils::effective_value(const dealii::SymmetricTensor<2, dim> &tensor) {
    auto dev = deviator(tensor);
    double dev_sq = dev * dev;
    return std::sqrt(1.5 * dev_sq);
}

} // namespace FEA
