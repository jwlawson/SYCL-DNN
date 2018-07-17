/*
 * Copyright 2018 Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SYCLDNN_INCLUDE_BACKEND_EIGEN_MATMUL_PROVIDER_H_
#define SYCLDNN_INCLUDE_BACKEND_EIGEN_MATMUL_PROVIDER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenMatmulProvider,
 * which provides single and batch matrix multiply implementations using Eigen.
 */

#include "sycldnn/backend/backend_traits.h"
#include "sycldnn/backend/crtp_backend.h"

namespace sycldnn {
namespace backend {

/**
 * Handler struct to provide matmul and batch_matmul implementations using
 * Eigen.
 *
 * This expects the Eigen Tensor module to have already been included. We don't
 * explicitly include it in this file so that the user has control of how Eigen
 * is included and which files are actually needed.
 */
template <typename EigenBackend>
struct EigenMatmulProvider
    : public CRTPBackend<EigenBackend, EigenMatmulProvider> {
  /**
   * Make TensorMap objects out of the provided pointers and dimensions, then
   * use Tensor Contraction to compute the matrix multiply.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event matmul(T const* const lhs, T const* const rhs,
                         T* const output, T const alpha, Index const m,
                         Index const k, Index const n) {
    static constexpr auto lhs_dim = TransposeLHS ? 0 : 1;
    static constexpr auto rhs_dim = TransposeRHS ? 1 : 0;
    using ConstTensorType = Eigen::Tensor<T const, 2, Eigen::RowMajor, Index>;
    using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
    using TensorType = Eigen::Tensor<T, 2, Eigen::RowMajor, Index>;
    using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
    using TensorShape = Eigen::DSizes<Index, 2>;
    using ContractDims =
        Eigen::IndexPairList<Eigen::type2indexpair<lhs_dim, rhs_dim>>;

    auto eigen_device = this->underlying_backend().get_eigen_device();

    TensorShape const lhs_shape{TransposeLHS ? k : m, TransposeLHS ? m : k};
    TensorShape const rhs_shape{TransposeRHS ? n : k, TransposeRHS ? k : n};
    TensorShape const out_shape{m, n};

    ConstTensor lhs_tensor{lhs, lhs_shape};
    ConstTensor rhs_tensor{rhs, rhs_shape};
    Tensor out_tensor{output, out_shape};

    if (alpha == static_cast<T>(0)) {
      out_tensor.device(eigen_device) =
          lhs_tensor.contract(rhs_tensor, ContractDims{});
    } else {
      out_tensor.device(eigen_device) =
          alpha * out_tensor + lhs_tensor.contract(rhs_tensor, ContractDims{});
    }
    // Eigen does not provide a way to access the SYCL event from kernels.
    return cl::sycl::event{};
  }

  /**
   * As Eigen Tensor does not have a batch matrix multiply, just fall back to
   * multiple calls to the standard matrix multiply.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event batch_matmul(T const* const lhs, T const* const rhs,
                               T* const output, Index const n_batches,
                               Index const m, Index const k, Index const n) {
    Index const lhs_size = m * k;
    Index const rhs_size = k * n;
    Index const out_size = m * n;

    cl::sycl::event event;
    for (int i = 0; i < n_batches; ++i) {
      Index const lhs_offset = lhs_size * i;
      Index const rhs_offset = rhs_size * i;
      Index const out_offset = out_size * i;
      event = matmul<TransposeLHS, TransposeRHS>(
          lhs + lhs_offset, rhs + rhs_offset, output + out_offset,
          static_cast<T>(0), m, k, n);
    }
    return event;
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_EIGEN_MATMUL_HANDLER_H_
