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
#ifndef SYCLDNN_SRC_HELPERS_MATH_H_
#define SYCLDNN_SRC_HELPERS_MATH_H_

#include "sycldnn/helpers/macros.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {
namespace math {
template <typename T>
static inline SNN_ALWAYS_INLINE T mad(T a, T b, T c) {
  return cl::sycl::mad(a, b, c);
}
/** Overload for 1-element vectors to workaround missing mad() function. */
template <typename T>
static inline SNN_ALWAYS_INLINE cl::sycl::vec<T, 1> mad(cl::sycl::vec<T, 1> a,
                                                        cl::sycl::vec<T, 1> b,
                                                        cl::sycl::vec<T, 1> c) {
  return cl::sycl::vec<T, 1>{cl::sycl::mad(a.s0(), b.s0(), c.s0())};
}
}  // namespace math
}  // namespace helpers
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_HELPERS_MATH_H_