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

#include <CL/sycl.hpp>

#include "src/pooling/queue_pooling_kernel_impl.h"
#include "sycldnn/accessor_types.h"
#include "sycldnn/internal/pooling/launch_internal.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace pooling {
namespace internal {

SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(int32_t, Average, Forward)
#ifdef SNN_USE_INT64
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(int64_t, Average, Forward)
#endif  // SNN_USE_INT64
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(float, Average, Forward)
#ifdef SNN_USE_HALF
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(cl::sycl::float, Average, Forward)
#endif  // SNN_USE_HALF
#ifdef SNN_USE_DOUBLE
SNN_INSTANTIATE_LAUNCH_POOLING_KERNEL(double, Average, Forward)
#endif  // SNN_USE_DOUBLE

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn
