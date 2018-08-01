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
#ifndef SYCLDNN_INCLUDE_CONV2D_IMPLEMENTATION_WINOGRAD_H_
#define SYCLDNN_INCLUDE_CONV2D_IMPLEMENTATION_WINOGRAD_H_

#include "sycldnn/conv2d/params.h"

#include "sycldnn/internal/conv2d/winograd/launch.h"

namespace sycldnn {
namespace conv2d {

/**
 * Launch the 2D convolution using the Winograd implementation.
 *
 * Will extract the SYCL buffers and SYCL queue from the backend and forward
 * these on to the precompiled kernels.
 *
 * \param input   Pointer to the input buffer
 * \param filter  Pointer to the filter buffer
 * \param output  Pointer to the output buffer
 * \param params  Convolution parameters
 * \param backend Backend to use to allocate temporary buffers and compute
 *                matrix multiplies
 * \return An SNNStatus containing the SYCL event tied to the kernel launch.
 */
template <typename T, typename ConvType, typename Backend>
inline SNNStatus launch_winograd(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    Conv2DParams const& params, Backend& backend) {
  return internal::winograd::launch<T, ConvType>(input, filter, output, params,
                                                 backend);
}

}  // namespace sycldnn
}  // namespace conv2d

#endif  // SYCLDNN_INCLUDE_CONV2D_IMPLEMENTATION_WINOGRAD_H_
