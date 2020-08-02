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
#ifndef SYCLDNN_SRC_MATMUL_BLOCKS_H_
#define SYCLDNN_SRC_MATMUL_BLOCKS_H_

#include "src/helpers/math.h"
#include "src/helpers/register_tile.h"
#include "src/helpers/vector_element.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

namespace sycldnn {
namespace matmul {

template <typename T, int Rows, int Cols>
struct VectorBlock final
    : public helpers::RegisterTile1D<
          typename helpers::VectorType<T, Cols>::type, Rows> {
  using VectorType = typename helpers::VectorType<T, Cols>::type;
  using helpers::RegisterTile1D<VectorType, Rows>::data;
};

template <typename T, int Rows, int Cols>
static VectorBlock<T, Cols, Rows> SNN_ALWAYS_INLINE
transpose_block(VectorBlock<T, Rows, Cols> const& input) {
  namespace vec_elem = helpers::vector_element;
  VectorBlock<T, Cols, Rows> output;
  SNN_PRAGMA_UNROLL
  for (int i = 0; i < Cols; ++i) {
    SNN_PRAGMA_UNROLL
    for (int j = 0; j < Rows; ++j) {
      vec_elem::set(output.data(i), j, vec_elem::get(input.data(j), i));
    }
  }
  return output;
}

template <typename VectorType, typename T,
          cl::sycl::access::address_space Space>
static VectorType SNN_ALWAYS_INLINE
load_row_unchecked(cl::sycl::multi_ptr<T, Space> row_start) {
  using VectorLoad = helpers::io::Load<VectorType>;
  VectorType output = VectorLoad()(row_start, 0);
  return output;
}

template <typename VectorType, int Cols, typename T,
          cl::sycl::access::address_space Space>
static VectorType SNN_ALWAYS_INLINE load_row_checked(
    cl::sycl::multi_ptr<T, Space> row_start, std::array<bool, Cols> mask) {
  namespace vec_elem = helpers::vector_element;
  using ScalarType = T;
  using ScalarLoad = helpers::io::Load<ScalarType>;
  VectorType output;
  if (mask[Cols - 1]) {
    output = load_row_unchecked<VectorType>(row_start);
  } else {
    for (int i = 0; i < Cols; ++i) {
      auto val = mask[i] ? ScalarLoad()(row_start, 0) : ScalarType{0};
      vec_elem::set(output, i, val);
      ++row_start;
    }
  }
  return output;
}

template <int Rows, int Cols, typename T, cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE load_block_checked(
    cl::sycl::multi_ptr<T const, Space> input, std::ptrdiff_t ld,
    std::array<bool, Rows> row_mask, std::array<bool, Cols> col_mask) {
  using OutputType = VectorBlock<T, Rows, Cols>;
  using VectorType = typename OutputType::VectorType;
  OutputType output;
  auto row_start_ptr = input;
  for (int i = 0; i < Rows; ++i) {
    if (row_mask[i]) {
      output.data(i) =
          load_row_checked<VectorType, Cols>(row_start_ptr, col_mask);
      row_start_ptr += ld;
    } else {
      output.data(i) = VectorType{0};
    }
  }
  return output;
}

template <int Rows, int Cols, bool Transpose, typename T,
          cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load_checked(cl::sycl::multi_ptr<T const, Space> input, std::ptrdiff_t ld,
             std::array<bool, Rows> row_mask, std::array<bool, Cols> col_mask) {
  VectorBlock<T, Rows, Cols> output;
  if (Transpose) {
    auto out_trans =
        load_block_checked<Cols, Rows>(input, ld, col_mask, row_mask);
    output = transpose_block(out_trans);
  } else {
    output = load_block_checked<Rows, Cols>(input, ld, row_mask, col_mask);
  }
  return output;
}

template <int Rows, int Cols, typename T, cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE load_block_unchecked(
    cl::sycl::multi_ptr<T const, Space> input, std::ptrdiff_t ld) {
  using OutputType = VectorBlock<T, Rows, Cols>;
  using VectorType = typename OutputType::VectorType;
  VectorBlock<T, Rows, Cols> output;
  auto row_start_ptr = input;
  for (int i = 0; i < Rows; ++i) {
    output.data(i) = load_row_unchecked<VectorType>(row_start_ptr);
    row_start_ptr += ld;
  }
  return output;
}

template <int Rows, int Cols, bool Transpose, typename T,
          cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load_unchecked(cl::sycl::multi_ptr<T const, Space> input, int ld) {
  VectorBlock<T, Rows, Cols> output;
  if (Transpose) {
    auto out_trans = load_block_unchecked<Cols, Rows>(input, ld);
    output = transpose_block(out_trans);
  } else {
    output = load_block_unchecked<Rows, Cols>(input, ld);
  }
  return output;
}

template <typename T, int Rows, int Cols>
static void SNN_ALWAYS_INLINE scalar_multiply(VectorBlock<T, Rows, Cols>& block,
                                              T val) {
  using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
  VectorType vector_val{val};
  for (int row = 0; row < Rows; ++row) {
    block.data(row) *= vector_val;
  }
}

template <bool TransposeLHS, bool TransposeRHS>
struct BlockMacc {};

template <>
struct BlockMacc<false, false> {
  template <typename T, int Rows, int Cols, int Acc>
  static void SNN_ALWAYS_INLINE run(VectorBlock<T, Rows, Acc> const& lhs,
                                    VectorBlock<T, Acc, Cols> const& rhs,
                                    VectorBlock<T, Rows, Cols>& accumulator) {
    using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
    namespace vec_elem = helpers::vector_element;
    for (int row = 0; row < Rows; ++row) {
      for (int acc = 0; acc < Acc; ++acc) {
        accumulator.data(row) =
            helpers::math::mad(VectorType{vec_elem::get(lhs.data(row), acc)},
                               rhs.data(acc), accumulator.data(row));
      }
    }
  }
};

template <>
struct BlockMacc<false, true> {
  template <typename T, int Rows, int Cols, int Acc>
  static void SNN_ALWAYS_INLINE run(VectorBlock<T, Rows, Acc> const& lhs,
                                    VectorBlock<T, Cols, Acc> const& rhs,
                                    VectorBlock<T, Rows, Cols>& accumulator) {
    namespace vec_elem = helpers::vector_element;
    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        // TODO(jwlawson): Provide accumulator/mad version of dot
        auto cur_val = vec_elem::get(accumulator.data(row), col);
        auto inc = helpers::math::dot(lhs.data(row), rhs.data(col));
        vec_elem::set(accumulator.data(row), col, cur_val + inc);
      }
    }
  }
};

template <>
struct BlockMacc<true, false> {
  template <typename T, int Rows, int Cols, int Acc>
  static void SNN_ALWAYS_INLINE run(VectorBlock<T, Acc, Rows> const& lhs,
                                    VectorBlock<T, Acc, Cols> const& rhs,
                                    VectorBlock<T, Rows, Cols>& accumulator) {
    using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
    namespace vec_elem = helpers::vector_element;
    for (int row = 0; row < Rows; ++row) {
      for (int acc = 0; acc < Acc; ++acc) {
        accumulator.data(row) =
            helpers::math::mad(VectorType{vec_elem::get(lhs.data(acc), row)},
                               rhs.data(acc), accumulator.data(row));
      }
    }
  }
};

template <>
struct BlockMacc<true, true> {
  template <typename T, int Rows, int Cols, int Acc>
  static void SNN_ALWAYS_INLINE run(VectorBlock<T, Acc, Rows> const& lhs,
                                    VectorBlock<T, Cols, Acc> const& rhs,
                                    VectorBlock<T, Rows, Cols>& accumulator) {
    namespace vec_elem = helpers::vector_element;
    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        for (int acc = 0; acc < Acc; ++acc) {
          auto cur_val = vec_elem::get(accumulator.data(row), col);
          auto lhs_val = vec_elem::get(lhs.data(acc), row);
          auto rhs_val = vec_elem::get(rhs.data(col), acc);
          auto next_val = helpers::math::mad(lhs_val, rhs_val, cur_val);
          vec_elem::set(accumulator.data(row), col, next_val);
        }
      }
    }
  }
};

template <bool TransposeLHS, bool TransposeRHS, typename T, int LRows,
          int LCols, int RRows, int RCols, int ARows, int ACols>
static void SNN_ALWAYS_INLINE
block_mmacc(VectorBlock<T, LRows, LCols> const& lhs,
            VectorBlock<T, RRows, RCols> const& rhs,
            VectorBlock<T, ARows, ACols>& accumulator) {
  BlockMacc<TransposeLHS, TransposeRHS>::run(lhs, rhs, accumulator);
}

template <int Cols, typename VectorType, typename T,
          cl::sycl::access::address_space Space>
static void SNN_ALWAYS_INLINE store_row(VectorType const& row_vec,
                                        cl::sycl::multi_ptr<T, Space> row_start,
                                        std::array<bool, Cols> valid_col) {
  using ScalarType = T;
  using ScalarStore = helpers::io::Store<ScalarType>;
  namespace vec_elem = helpers::vector_element;
  for (int i = 0; i < Cols; ++i) {
    if (valid_col[i]) {
      ScalarStore()(row_start, 0, vec_elem::get(row_vec, i));
      ++row_start;
    }
  }
}

template <int Rows, int Cols, typename T, cl::sycl::access::address_space Space>
static void SNN_ALWAYS_INLINE store_block(
    VectorBlock<T, Rows, Cols> const& block,
    cl::sycl::multi_ptr<T, Space> output, int ld,
    std::array<bool, Rows> valid_row, std::array<bool, Cols> valid_col) {
  auto row_start_ptr = output;
  for (int i = 0; i < Rows; ++i) {
    if (valid_row[i]) {
      store_row<Cols>(block.data(i), row_start_ptr, valid_col);
      row_start_ptr += ld;
    }
  }
}

template <int Rows, int Cols, typename T, cl::sycl::access::address_space Space>
static void SNN_ALWAYS_INLINE
store_block_unchecked(VectorBlock<T, Rows, Cols> const& block,
                      cl::sycl::multi_ptr<T, Space> output, int ld) {
  using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
  using VectorStore = helpers::io::Store<VectorType>;
  for (int i = 0; i < Rows; ++i) {
    VectorStore()(output, 0, block.data(i));
    output += ld;
  }
}

}  // namespace matmul
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_MATMUL_BLOCKS
