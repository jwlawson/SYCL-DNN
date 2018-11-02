/*
 * Copyright 2018 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#ifndef SYCLDNN_BENCH_CONV2D_ARM_CONV2D_EXECUTOR_H_
#define SYCLDNN_BENCH_CONV2D_ARM_CONV2D_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"

#include "bench/fixture/base_executor.h"

// The OpenCL C++ wrapper, used by ARM Compute Library, generates warnings
// about deprecated functions. This silences those warnings.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#pragma GCC diagnostic pop

namespace sycldnn {
namespace bench {

namespace arm = arm_compute;

/** Executor to perform the Conv2d benchmark using ARM Compute Library.  */
template <typename Benchmark, typename ConvType>
struct ARMConv2DExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using Conv2DParams = conv2d::Conv2DParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a conv2d benchmark with the given parameters. */
  void execute(State& state, Conv2DParams const& params) {
    auto& scheduler = arm::CLScheduler::get();
    auto context = cl::Context::getDefault();
    auto queue = cl::CommandQueue::getDefault();
    auto device = cl::Device::getDefault();
    scheduler.init(context, queue, device);

    // Allocate tensors.
    arm::CLTensor W, X, Z, B;
    X.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.in_rows, params.in_cols,
                                         params.channels, params.batch),
                        arm::Format::F32));
    Z.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.out_rows, params.out_cols,
                                         params.features, params.batch),
                        arm::Format::F32));
    W.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.window_rows, params.window_cols,
                                         params.channels, params.features),
                        arm::Format::F32));
    B.allocator()->init(
        arm::TensorInfo(arm::TensorShape(params.features), arm::Format::F32));

    // Construct a convolution layer.
    arm::CLConvolutionLayer conv1;
    arm::PadStrideInfo psi(params.stride_cols, params.stride_rows,
                           params.pad_cols, params.pad_rows);
    conv1.configure(&X, &W, &B, &Z, psi);

    // Validate the configuration.
    auto status = conv1.validate(X.info(), W.info(), B.info(), Z.info(), psi);
    if (!status) {
      state.SkipWithError(status.error_description().c_str());
      return;
    }

    // Allocate the tensors themselves.
    X.allocator()->allocate();
    Z.allocator()->allocate();
    W.allocator()->allocate();
    B.allocator()->allocate();

    // Run the layer once to eliminate lazy behaviour.
    conv1.run();
    scheduler.sync();

    for (auto _ : state) {
      this->start_timing();
      conv1.run();
      scheduler.sync();
      this->end_timing();

      this->set_iteration_time(state);
    }

    X.allocator()->free();
    Z.allocator()->free();
    W.allocator()->free();
    B.allocator()->free();

    auto& benchmark = underlying_benchmark();
    benchmark.template set_items_processed<ConvType>(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(
        state, sycldnn::conv2d::get_sizes<ConvType>(params));
    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_CONV2D_ARM_CONV2D_EXECUTOR_H_
