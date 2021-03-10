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

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/clblast_backend.h"
#include "sycldnn/backend/clblas_backend.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/sizes.h"
#include "sycldnn/conv2d/workspace_size.h"

#include "sycldnn/helpers/padding.h"
#include "sycldnn/helpers/ratio.h"

#include "sycldnn/pointwise/launch.h"

#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"

#include "sycldnn/transpose/launch.h"

#include "sycldnn/padding_mode.h"
#include "sycldnn/status.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <istream>
#include <memory>
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

namespace snn = sycldnn;
namespace sycl = cl::sycl;

//using Backend = snn::backend::SyclBLASBackend;
using Backend = snn::backend::SNNBackend;
//using Backend = snn::backend::CLBlastBackend;
//using Backend = snn::backend::clBLASBackend;

using DeviceMem = Backend::pointer_type<float>;

class BiasAdd;
class SoftMax;

// Shortcut functions to create parameters suitable for VGG specifically
snn::conv2d::Conv2DParams make_3x3_conv_params(int channels, int features,
                                               int input) {
  snn::conv2d::Conv2DParams p = {channels, features, 1, input, input, 3, 3,
                                 1,        1,        0, 0,     0,     0};
  p = snn::helpers::add_padding_to(p, snn::PaddingMode::SAME);
  return p;
}

snn::pooling::PoolingParams make_2x2_pooling_params(int channels, int input) {
  snn::pooling::PoolingParams p = {input, input, 0, 0,        2, 2,
                                   2,     2,     1, channels, 0, 0};
  p = snn::helpers::add_padding_to(p, snn::PaddingMode::VALID);
  return p;
}

// Helper function that reads binary data produced by h5tobin.py into a vector
std::vector<char> read_binary_data(std::string const& name) {
  std::ifstream file(name, std::ios_base::binary | std::ios_base::in);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file " + name);
  }
  std::vector<char> output{std::istreambuf_iterator<char>{file}, {}};
  return output;
}

// Base class of all layer types to present unified interface and construction
struct Layer {
  Backend& backend_;

  Layer(Backend& b) : backend_(b) {}
  virtual ~Layer() = default;

  virtual DeviceMem get_output() const = 0;
  virtual size_t get_output_size() const = 0;
  virtual snn::SNNStatus run() = 0;
};

struct ConvolutionLayer : Layer {
  snn::conv2d::Conv2DParams params_;
  snn::conv2d::ConvSizes sizes_;
  DeviceMem input_;
  DeviceMem filter_;
  DeviceMem output_;
  DeviceMem& workspace_;
  size_t workspace_size_;
  snn::conv2d::Selector& selector_;

  // Sets parameters and copies data into filter buffer
  ConvolutionLayer(snn::conv2d::Conv2DParams params, DeviceMem prev,
                   DeviceMem& workspace, size_t workspace_size,
                   std::string const& name, Backend& b,
                   snn::conv2d::Selector& selector)
      : Layer(b),
        params_{params},
        sizes_{
            snn::conv2d::get_sizes<snn::conv2d::conv_type::Forward>(params_)},
        input_{prev},
        filter_{b.allocate<float>(sizes_.filter_size)},
        output_{b.allocate<float>(sizes_.output_size)},
        workspace_{workspace},
        workspace_size_{workspace_size},
        selector_{selector} {
    std::vector<char> filter = read_binary_data(name);
    assert(filter.size() == sizes_.filter_size * sizeof(float));
    auto data_size = sycl::range<1>{sizes_.filter_size};
    auto char_buf =
        filter_.get_buffer().reinterpret<char>(data_size * sizeof(float));
    auto queue = backend_.get_queue();
    auto copy_event = queue.submit([&](sycl::handler& h) {
      auto acc = char_buf.get_access<sycl::access::mode::discard_write>(h);
      h.copy(filter.data(), acc);
    });
    copy_event.wait_and_throw();
  }

  DeviceMem get_output() const override { return output_; }
  size_t get_output_size() const override { return sizes_.output_size; }

  snn::SNNStatus run() override {
    return snn::conv2d::launch<float, snn::conv2d::conv_type::Forward>(
        input_, filter_, output_, params_, selector_, backend_, workspace_,
        workspace_size_);
  }
};

struct PoolingLayer : Layer {
  snn::pooling::PoolingParams params_;
  snn::pooling::PoolingSizes sizes_;
  DeviceMem input_;
  DeviceMem output_;

  PoolingLayer(snn::pooling::PoolingParams params, DeviceMem prev, Backend& b)
      : Layer(b),
        params_{params},
        sizes_{snn::pooling::get_sizes<snn::pooling::Forward>(params_)},
        input_{prev},
        output_{b.allocate<float>(sizes_.output_size)} {}

  DeviceMem get_output() const override { return output_; }
  size_t get_output_size() const override { return sizes_.output_size; }

  snn::SNNStatus run() override {
    return snn::pooling::launch<float, snn::pooling::Max,
                                snn::pooling::Forward>(input_, output_, params_,
                                                       backend_);
  }
};

struct ReLULayer : Layer {
  size_t size_;
  DeviceMem input_;
  DeviceMem output_;

  ReLULayer(size_t size, DeviceMem prev, Backend& b)
      : Layer{b}, size_{size}, input_{prev}, output_{b.allocate<float>(size)} {}

  DeviceMem get_output() const override { return output_; }
  size_t get_output_size() const override { return size_; }

  snn::SNNStatus run() override {
    return snn::pointwise::launch<float, snn::pointwise::Relu,
                                  snn::pointwise::Forward>(input_, output_,
                                                           size_, backend_);
  }
};

struct FullyConnectedLayer : Layer {
  size_t weights_cols_;
  size_t output_size_;
  DeviceMem input_;
  DeviceMem weights_;
  DeviceMem output_;

  FullyConnectedLayer(size_t input, size_t output, DeviceMem prev,
                      std::string const& name, Backend& b)
      : Layer{b},
        weights_cols_{input},
        output_size_{output},
        input_{prev},
        weights_{b.allocate<float>(input * output)},
        output_{b.allocate<float>(output)} {
    std::vector<char> weights = read_binary_data(name);
    assert(weights.size() == weights_cols_ * output_size_ * sizeof(float));
    auto data_size = sycl::range<1>{weights_cols_ * output_size_};
    auto char_buf =
        weights_.get_buffer().reinterpret<char>(data_size * sizeof(float));
    auto queue = backend_.get_queue();
    auto copy_event = queue.submit([&](sycl::handler& h) {
      auto acc = char_buf.get_access<sycl::access::mode::discard_write>(h);
      h.copy(weights.data(), acc);
    });
    // keeps weights alive and accessible
    copy_event.wait_and_throw();
  }

  DeviceMem get_output() const override { return output_; }
  size_t get_output_size() const override { return output_size_; }
  snn::SNNStatus run() override {
    using const_DeviceMem = Backend::internal_pointer_type<float const>;
    return {backend_.matmul<false, false>(
                const_DeviceMem{input_}, const_DeviceMem{weights_}, output_,
                0.f, 1u, static_cast<unsigned>(weights_cols_),
                static_cast<unsigned>(output_size_)),
            snn::StatusCode::OK};
  }
};

struct BiasAddLayer : Layer {
  size_t size_;  // WxH
  size_t features_;
  DeviceMem input_;
  DeviceMem biases_;
  DeviceMem output_;

  BiasAddLayer(size_t size, size_t features, DeviceMem prev,
               std::string const& name, Backend& b)
      : Layer{b},
        size_{size / features},
        features_{features},
        input_{prev},
        biases_{b.allocate<float>(features)},
        output_{b.allocate<float>(size_ * features_)} {
    std::vector<char> biases = read_binary_data(name);
    assert(biases.size() == features * sizeof(float));
    auto data_size = sycl::range<1>{features};
    auto buf = biases_.get_buffer();
    auto char_buf = buf.reinterpret<char>(data_size * sizeof(float));
    auto queue = backend_.get_queue();
    auto copy_event = queue.submit([&](sycl::handler& h) {
      auto acc = char_buf.get_access<sycl::access::mode::discard_write>(h);
      h.copy(biases.data(), acc);
    });
    copy_event.wait_and_throw();
  }

  DeviceMem get_output() const override { return output_; }
  size_t get_output_size() const override { return size_ * features_; }

  snn::SNNStatus run() override {
    snn::SNNStatus status;
    status.event = backend_.get_queue().submit([&](sycl::handler& cgh) {
      auto in = backend_.get_mem_object(input_, size_*features_).read_accessor(cgh).get_accessor();
      auto out = backend_.get_mem_object(output_, size_*features_).write_accessor(cgh).get_accessor();
      auto biases = backend_.get_mem_object(biases_, features_).read_accessor(cgh).get_accessor();
      auto range = sycl::range<2>{size_, features_};
      auto features = features_;
      cgh.parallel_for<BiasAdd>(range, [=](sycl::item<2> item) {
        auto feat = item.get_id(1);
        auto id = item.get_id(0) * features + feat;
        out[id] = in[id] + biases[feat];
      });
    });
    status.status = snn::StatusCode::OK;
    return status;
  }
};

// Only does part transform on device, reduction of 1k elems is fine on host
struct SoftmaxLayer : Layer {
  size_t output_size_;
  DeviceMem input_;
  std::vector<float>& output_;

  SoftmaxLayer(size_t output_size, DeviceMem prev, std::vector<float>& output,
               Backend& b)
      : Layer{b}, output_size_{output_size}, input_{prev}, output_{output} {}

  DeviceMem get_output() const override { return input_; }
  size_t get_output_size() const override { return output_size_; }

  snn::SNNStatus run() override {
    auto r = sycl::range<1>{output_size_};
    output_.resize(output_size_);
    sycl::buffer<float> buf{r};
    auto q = backend_.get_queue();
    auto ev = q.submit([&](sycl::handler& h) {
      auto in = backend_.get_mem_object(input_, output_size_).read_accessor(h).get_accessor();
      auto out = buf.get_access<sycl::access::mode::write>(h);
      auto os = output_size_;
      auto kernel_range = sycl::range<1>{
          sycldnn::helpers::round_up_to_nearest_multiple(r[0], 64)};
      h.parallel_for<SoftMax>(kernel_range, [=](sycl::item<1> i) {
        auto id = i.get_id(0);
        if (id < os) {
          out[id] = sycl::exp(in[id]);
        }
      });
    });

    ev = q.submit([&](sycl::codeplay::host_handler& h) {
      auto data = buf.get_access<sycl::access::mode::read>();
      h.host_task([ =, out = output_.begin() ]() mutable {
        auto start = data.get_pointer();
        auto finish = start + data.get_count();
        auto sum = std::accumulate(start, finish, 0.f);
        std::transform(start, finish, out, [sum](float x) { return x / sum; });
      });
    });
    return {ev, snn::StatusCode::OK};
  }
};

class Network {
  std::vector<std::unique_ptr<Layer>> network_;
  DeviceMem input_;
  DeviceMem workspace_;  // Temporary buffer used in some convolution types
  size_t workspace_max_;
  std::vector<float>& output_;
  unsigned int layer_number_;  // Used to read data correctly
  std::string data_dir_;       // Location of network weights
  Backend& backend_;
  snn::conv2d::Selector& selector_;

 public:
  Network(DeviceMem input, std::vector<float>& output, std::string const& dir,
          Backend& backend, snn::conv2d::Selector& sel)
      : network_{},
        input_{input},
        workspace_{},
        workspace_max_{0},
        output_{output},
        layer_number_{1},
        data_dir_{dir},
        backend_{backend},
        selector_{sel} {}

  // Layers are their own types, number of parameters differs between each
  template <typename LayerT, int dims>
  void add_layer(std::array<size_t, dims>);

  // Runs each layer, checks for exceptions after every layer
  snn::SNNStatus test() {
    workspace_ = backend_.allocate<float>(workspace_max_ * sizeof(float));
    snn::SNNStatus status;
    for (auto& layer : network_) {
      status = layer->run();
      status.event.wait_and_throw();
    }
    return status;
  }

  snn::SNNStatus run() {
    if (workspace_.get_buffer().get_count() < workspace_max_) {
      workspace_ = backend_.allocate<float>(workspace_max_ * sizeof(float));
    }
    snn::SNNStatus status;
    for (auto& layer : network_) {
      status = layer->run();
    }
    return status;
  }
};

template <>
void Network::add_layer<ConvolutionLayer, 3>(std::array<size_t, 3> sizes) {
  auto params = make_3x3_conv_params(sizes[0], sizes[1], sizes[2]);
  auto new_size =
      snn::conv2d::query_workspace_size<snn::conv2d::conv_type::Forward>(
          params, selector_)
          .recommended_size;
  workspace_max_ = new_size > workspace_max_ ? new_size : workspace_max_;
  auto input = network_.empty() ? input_ : network_.back()->get_output();
  auto layer_name =
      data_dir_ + "/layer_" + std::to_string(layer_number_) + "-weights.bin";

  network_.emplace_back(new ConvolutionLayer(params, input, workspace_,
                                             workspace_max_, layer_name,
                                             backend_, selector_));
}

template <>
void Network::add_layer<PoolingLayer, 2>(std::array<size_t, 2> sizes) {
  auto params = make_2x2_pooling_params(sizes[0], sizes[1]);
  network_.emplace_back(
      new PoolingLayer(params, network_.back()->get_output(), backend_));
}

template <>
void Network::add_layer<ReLULayer, 0>(std::array<size_t, 0>) {
  network_.emplace_back(new ReLULayer(network_.back()->get_output_size(),
                                      network_.back()->get_output(), backend_));
}

template <>
void Network::add_layer<FullyConnectedLayer, 1>(std::array<size_t, 1> sizes) {
  auto layer_name =
      data_dir_ + "/layer_" + std::to_string(layer_number_) + "-weights.bin";
  network_.emplace_back(new FullyConnectedLayer(
      network_.back()->get_output_size(), sizes[0],
      network_.back()->get_output(), layer_name, backend_));
}

template <>
void Network::add_layer<BiasAddLayer, 1>(std::array<size_t, 1> sizes) {
  auto layer_name =
      data_dir_ + "/layer_" + std::to_string(layer_number_++) + "-biases.bin";
  network_.emplace_back(
      new BiasAddLayer(network_.back()->get_output_size(), sizes[0],
                       network_.back()->get_output(), layer_name, backend_));
}

template <>
void Network::add_layer<SoftmaxLayer, 0>(std::array<size_t, 0>) {
  network_.emplace_back(new SoftmaxLayer(network_.back()->get_output_size(),
                                         network_.back()->get_output(), output_,
                                         backend_));
}

DeviceMem read_image_data(std::string const& name, Backend& backend) {
  sycl::range<1> r{224 * 224 * 3};  // vgg input size
  sycl::buffer<float> b{r};
  auto data = read_binary_data(name);
  assert(data.size() == 224 * 224 * 3 * sizeof(float));
  {
    auto char_data = b.reinterpret<char>(r * sizeof(float));
    auto event = backend.get_queue().submit([&](sycl::handler& cgh) {
      auto acc = char_data.get_access<sycl::access::mode::discard_write>(cgh);
      cgh.copy(data.data(), acc);
    });
    event.wait_and_throw();
  }
  return DeviceMem{b, 0};
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "USAGE: vgg <directory> <image>\n";
    return 1;
  }

  sycl::queue q([](sycl::exception_list l) {
    for (auto e : l) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception& e) {
        std::cout << e.what() << " " << e.get_cl_code() << "\n";
      }
    }
  });
  Backend backend(q);
  auto selector = snn::conv2d::get_default_selector(q.get_device());
  // This section encodes the size of each layer.
  std::vector<float> output;
  std::string data_dir{argv[1]};
  auto input = read_image_data(argv[2], backend);
  Network network(input, output, data_dir, backend, *selector);
  network.add_layer<ConvolutionLayer, 3>({3, 64, 224});
  network.add_layer<BiasAddLayer, 1>({64});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({64, 64, 224});
  network.add_layer<BiasAddLayer, 1>({64});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<PoolingLayer, 2>({64, 224});
  network.add_layer<ConvolutionLayer, 3>({64, 128, 112});
  network.add_layer<BiasAddLayer, 1>({128});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({128, 128, 112});
  network.add_layer<BiasAddLayer, 1>({128});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<PoolingLayer, 2>({128, 112});
  network.add_layer<ConvolutionLayer, 3>({128, 256, 56});
  network.add_layer<BiasAddLayer, 1>({256});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({256, 256, 56});
  network.add_layer<BiasAddLayer, 1>({256});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({256, 256, 56});
  network.add_layer<BiasAddLayer, 1>({256});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<PoolingLayer, 2>({256, 56});
  network.add_layer<ConvolutionLayer, 3>({256, 512, 28});
  network.add_layer<BiasAddLayer, 1>({512});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({512, 512, 28});
  network.add_layer<BiasAddLayer, 1>({512});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({512, 512, 28});
  network.add_layer<BiasAddLayer, 1>({512});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<PoolingLayer, 2>({512, 28});
  network.add_layer<ConvolutionLayer, 3>({512, 512, 14});
  network.add_layer<BiasAddLayer, 1>({512});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({512, 512, 14});
  network.add_layer<BiasAddLayer, 1>({512});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<ConvolutionLayer, 3>({512, 512, 14});
  network.add_layer<BiasAddLayer, 1>({512});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<PoolingLayer, 2>({512, 14});
  network.add_layer<FullyConnectedLayer, 1>({4096});
  network.add_layer<BiasAddLayer, 1>({4096});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<FullyConnectedLayer, 1>({4096});
  network.add_layer<BiasAddLayer, 1>({4096});
  network.add_layer<ReLULayer, 0>({});
  network.add_layer<FullyConnectedLayer, 1>({1000});
  network.add_layer<BiasAddLayer, 1>({1000});
  network.add_layer<SoftmaxLayer, 0>({});

  network.test();
  auto index = std::max_element(output.begin(), output.end());
  std::cout << "classed as " << std::distance(output.begin(), index)
            << ", value " << (index != std::end(output) ? *index : 0.f)
            << std::endl;

  int loops = 8;
  do {
    auto st = std::chrono::high_resolution_clock::now();
    auto status = network.run();
    status.event.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    auto milli = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - st);
    std::cout << std::setprecision(4) << milli.count() << " ms\n";
  } while (--loops);

  q.wait_and_throw();
  return 0;
}
