/*
 * Copyright Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_BACKEND_CLBLAS_BACKEND_H_
#define SYCLDNN_INCLUDE_BACKEND_CLBLAS_BACKEND_H_

/**
 * \file
 * Contains the implementation of the \ref sycldnn::backend::clBLASBackend,
 * which allocates memory via SYCL and does GEMM and GEMV operations with
 * the clBLAS library.
 */
#include "sycldnn/backend/backend_traits.h"
#include "sycldnn/backend/device_mem_pointer.h"
#include "sycldnn/backend/opencl_backend.h"
#include "sycldnn/helpers/macros.h"

#include "sycldnn/mem_object.h"

#include <CL/cl.h>
#include <clBLAS.h>
#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

namespace sycldnn {
namespace backend {

/**
 * \brief clBLAS backend for SYCL-DNN.
 *
 * Provides pointer handling and matrix multiplies using clBLAS.
 */
class clBLASBackend final : public OpenCLBackend {
 public:
  using OpenCLBackend::internal_pointer_type;
  using OpenCLBackend::pointer_type;
  /** Constructor initializes the clBLAS library. */
  clBLASBackend(cl::sycl::queue& queue) : OpenCLBackend{queue} {
    clblasSetup();
  }

  /** Explicit destructor tears down the clBLAS library. */
  ~clBLASBackend() { clblasTeardown(); }

  /**
   * Gets a descriptive name for this backend.
   * \return a descriptive name for this backend.
   */
  char const* name() const { return "clBLASBackend"; }

  /**
   * A wrapper around a call to GEMM.
   *
   * Should perform the matrix multiply operation:
   *   output = lhs * rhs + beta * output
   * where lhs is a [m x k] matrix, rhs is a [k x n] matrix. The `bool`
   * template parameters determine whether or not to transpose the matrices.
   *
   * \param [in]     lhs       Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs       Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output    Pointer to a buffer containing the output matrix.
   * \param [in]     beta      Scale multiplier for the output matrix.
   * \param [in]     m         Number of rows in the LHS matrix.
   * \param [in]     k         Number of columns in the LHS matrix and rows in
   *                           the RHS matrix.
   * \param [in]     n         Number of columns in the RHS matrix.
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event matmul(internal_pointer_type<const T> const lhs,
                         internal_pointer_type<const T> const rhs,
                         internal_pointer_type<T> const output, T const beta,
                         Index const m, Index const k, Index const n) {
    static_assert(
        std::is_same<T, float>::value,
        "clBLAS support is currently only available for float data type");
    auto a_buf = lhs.get_buffer();
    auto b_buf = rhs.get_buffer();
    auto o_buf = output.get_buffer();
    auto a_offset = lhs.get_offset();
    auto b_offset = rhs.get_offset();
    auto o_offset = output.get_offset();

    auto ev = queue_.submit([&](cl::sycl::codeplay::handler& cgh) {
      using namespace cl::sycl::access;
      auto a_acc = a_buf.template get_access<mode::read>(cgh);
      auto b_acc = b_buf.template get_access<mode::read>(cgh);
      auto o_acc = o_buf.template get_access<mode::read_write>(cgh);

      cgh.interop_task([=](cl::sycl::codeplay::interop_handle const& han) {
        auto a = han.get(a_acc);
        auto b = han.get(b_acc);
        auto o = han.get(o_acc);

        auto transa = TransposeLHS ? clblasTrans : clblasNoTrans;
        auto transb = TransposeRHS ? clblasTrans : clblasNoTrans;

        constexpr T alpha = static_cast<T>(1);

        cl_event e;
        clblasStatus code;
        constexpr size_t num_queues = 1;
        constexpr size_t num_waits = 0;
        if (m == 1) {
          // The LHS matrix is actually a vector. Switch and transpose the
          // matrices.
          auto gemv_m = TransposeRHS ? n : k;
          auto gemv_n = TransposeRHS ? k : n;
          auto gemv_ldb = gemv_n;
          auto gemv_transb = TransposeRHS ? clblasNoTrans : clblasTrans;
          constexpr size_t increment = 1;
          code = clblasSgemv(clblasRowMajor, gemv_transb, gemv_m, gemv_n, alpha,
                             b, b_offset, gemv_ldb, a, a_offset, increment,
                             beta, o, o_offset, increment, num_queues,
                             &cl_queue_, num_waits, nullptr, &e);
        } else if (n == 1) {
          // The RHS matrix is actually a vector
          auto gemv_m = TransposeLHS ? k : m;
          auto gemv_n = TransposeLHS ? m : k;
          auto gemv_lda = gemv_n;
          constexpr size_t increment = 1;
          code = clblasSgemv(clblasRowMajor, transa, gemv_m, gemv_n, alpha, a,
                             a_offset, gemv_lda, b, b_offset, increment, beta,
                             o, o_offset, increment, num_queues, &cl_queue_,
                             num_waits, nullptr, &e);
        } else {
          auto lda = TransposeLHS ? m : k;
          auto ldb = TransposeRHS ? k : n;
          auto ldc = n;

          code =
              clblasSgemm(clblasRowMajor, transa, transb, m, n, k, alpha, a,
                          a_offset, lda, b, b_offset, ldb, beta, o, o_offset,
                          ldc, num_queues, &cl_queue_, num_waits, nullptr, &e);
        }
        if (code != clblasSuccess) {
          std::string excep("Bad return code from clBLAS GEMM: ");
          throw std::runtime_error(excep +
                                   std::to_string(static_cast<int>(code)));
        }
        clWaitForEvents(1, &e);
      });
    });
    return ev;
  }

  /**
   * Compute a batch of matrix multiplies.
   *
   * Assumes that lhs is a [batch x m x k] tensor and rhs is a [batch x k x n]
   * tensor.
   * Should perform the batched matrix multiply operation:
   *   output[i] = lhs[i] * rhs[i]
   * for 0 <= i < batch. Each matrix is assumed to be contiguous in memory and
   * in row-major format. The `bool` template parameters determine whether or
   * not to transpose the matrices.
   *
   * \param [in]     lhs       Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs       Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output    Pointer to a buffer containing the output matrix.
   * \param [in]     n_batches The number of matrices in the batch.
   * \param [in]     m         Number of rows in the LHS matrix.
   * \param [in]     k         Number of columns in the LHS matrix and rows in
   *                           the RHS matrix.
   * \param [in]     n         Number of columns in the RHS matrix.
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event batch_matmul(internal_pointer_type<const T> const lhs,
                               internal_pointer_type<const T> const rhs,
                               internal_pointer_type<T> const output,
                               Index const n_batches, Index const m,
                               Index const k, Index const n) {
    auto a_buf = lhs.get_buffer();
    auto b_buf = rhs.get_buffer();
    auto o_buf = output.get_buffer();
    auto init_a_offset = lhs.get_offset();
    auto init_b_offset = rhs.get_offset();
    auto init_o_offset = output.get_offset();

    auto ev = queue_.submit([&](cl::sycl::codeplay::handler& cgh) {
      using namespace cl::sycl::access;
      auto a_acc = a_buf.template get_access<mode::read>(cgh);
      auto b_acc = b_buf.template get_access<mode::read>(cgh);
      auto o_acc = o_buf.template get_access<mode::read_write>(cgh);

      cgh.interop_task([=](cl::sycl::codeplay::interop_handle const& han) {
        auto a = han.get(a_acc);
        auto b = han.get(b_acc);
        auto o = han.get(o_acc);

        auto transa = TransposeLHS ? clblasTrans : clblasNoTrans;
        auto transb = TransposeRHS ? clblasTrans : clblasNoTrans;

        constexpr T alpha = static_cast<T>(1);
        constexpr T beta = static_cast<T>(0);

        std::vector<cl_event> e(n_batches);
        Index const lhs_size = m * k;
        Index const rhs_size = k * n;
        Index const out_size = m * n;
        for (int i = 0; i < n_batches; ++i) {
          Index const a_offset = init_a_offset + lhs_size * i;
          Index const b_offset = init_b_offset + rhs_size * i;
          Index const o_offset = init_o_offset + out_size * i;
          clblasStatus code;
          constexpr size_t num_queues = 1;
          constexpr size_t num_waits = 0;
          if (m == 1) {
            // The LHS matrix is actually a vector. Switch and transpose the
            // matrices.
            auto gemv_m = TransposeRHS ? n : k;
            auto gemv_n = TransposeRHS ? k : n;
            auto gemv_ldb = gemv_n;
            auto gemv_transb = TransposeRHS ? clblasNoTrans : clblasTrans;
            constexpr size_t increment = 1;
            code = clblasSgemv(
                clblasRowMajor, gemv_transb, gemv_m, gemv_n, alpha, b, b_offset,
                gemv_ldb, a, a_offset, increment, beta, o, o_offset, increment,
                num_queues, &cl_queue_, num_waits, nullptr, &e[i]);
          } else if (n == 1) {
            // The RHS matrix is actually a vector
            auto gemv_m = TransposeLHS ? k : m;
            auto gemv_n = TransposeLHS ? m : k;
            auto gemv_lda = gemv_n;
            constexpr size_t increment = 1;
            code = clblasSgemv(clblasRowMajor, transa, gemv_m, gemv_n, alpha, a,
                               a_offset, gemv_lda, b, b_offset, increment, beta,
                               o, o_offset, increment, num_queues, &cl_queue_,
                               num_waits, nullptr, &e[i]);
          } else {
            auto lda = TransposeLHS ? m : k;
            auto ldb = TransposeRHS ? k : n;
            auto ldc = n;

            code = clblasSgemm(clblasRowMajor, transa, transb, m, n, k, alpha,
                               a, a_offset, lda, b, b_offset, ldb, beta, o,
                               o_offset, ldc, num_queues, &cl_queue_, num_waits,
                               nullptr, &e[i]);
          }
          if (code != clblasSuccess) {
            std::string excep("Bad return code from clBLAS GEMM: ");
            throw std::runtime_error(excep +
                                     std::to_string(static_cast<int>(code)));
          }
        }
        clWaitForEvents(n_batches, &e[0]);
      });
    });
    return ev;
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_SYCL_BLAS_BACKEND_H_
