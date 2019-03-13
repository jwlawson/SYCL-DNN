/*
 * Copyright 2019 Codeplay Software Ltd.
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
#include "arm_fixture.h"
#include "mobilenet_param_set.h"

#define MOBILENET_BENCHMARK(N, WIN, STR, C, W, H, F)                       \
  CONVOLUTION_BENCHMARK("MobileNet",                                       \
                        ARM_Forward_##N##_##C##_##W##_##H##_##WIN##_##STR, \
                        ParameterSet<N, WIN, STR, C, W, H, F>,             \
                        sycldnn::conv2d::conv_type::Forward)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define MOBILENET_PARAMS(WIN, STR, C, W, H, F) \
  MOBILENET_BENCHMARK(1, WIN, STR, C, W, H, F);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(WIN, STR, C, W, H, F) \
  MOBILENET_BENCHMARK(4, WIN, STR, C, W, H, F);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(WIN, STR, C, W, H, F) \
  MOBILENET_BENCHMARK(32, WIN, STR, C, W, H, F);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define MOBILENET_PARAMS(WIN, STR, C, W, H, F) \
  MOBILENET_BENCHMARK(2, WIN, STR, C, W, H, F);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(WIN, STR, C, W, H, F) \
  MOBILENET_BENCHMARK(8, WIN, STR, C, W, H, F);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(WIN, STR, C, W, H, F) \
  MOBILENET_BENCHMARK(16, WIN, STR, C, W, H, F);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(WIN, STR, C, W, H, F) \
  MOBILENET_BENCHMARK(64, WIN, STR, C, W, H, F);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
