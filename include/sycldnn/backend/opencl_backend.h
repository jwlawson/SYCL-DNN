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
#ifndef SYCLDNN_INCLUDE_BACKEND_OPENCL_BACKEND_H_
#define SYCLDNN_INCLUDE_BACKEND_OPENCL_BACKEND_H_

/**
 * \file
 * Contains the implementation of the base class \ref sycldnn::backend::OpenCLBackend,
 * which handles memory via SYCL and provides helpers to access the cl_mem objects.
 * Should be extended to provide the matmul and batch_matmul functions.
 */
#include "sycldnn/backend/backend_traits.h"
#include "sycldnn/backend/device_mem_pointer.h"
#include "sycldnn/helpers/macros.h"

#include "sycldnn/mem_object.h"

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

namespace sycldnn {
namespace backend {

// Forward declaration to allow the BackendTraits specialisation.
class OpenCLBackend;

/** Specialisation of \ref BackendTraits for the clBLAS backend. */
template <>
struct BackendTraits<OpenCLBackend> {
  /** External pointer type exposed by clBLASBackend - same as internal. */
  template <typename T>
  using pointer_type = DeviceMemPointer<T>;

  /** Internal pointer type used in clBLASBackend - same as external. */
  template <typename T>
  using internal_pointer_type = DeviceMemPointer<T>;
};

/**
 * \brief OpenCL base backend for SYCL-DNN.
 *
 * Provides pointer handling usign SYCL and provides acces to underlying cl_mem
 * objects.
 */
class OpenCLBackend {
  protected:
  /** Copy of SYCL queue that wraps the cl_command_queue. */
  cl::sycl::queue queue_;
  /** Cached OpenCL command queue from SYCL queue. */
  cl_command_queue cl_queue_;

 public:
  /** The pointer type used in interface of the clBLASBackend. */
  template <typename T>
  using pointer_type =
      typename BackendTraits<OpenCLBackend>::template pointer_type<T>;
  /** The internal pointer type used internally by the clBLASBackend. */
  template <typename T>
  using internal_pointer_type =
      typename BackendTraits<OpenCLBackend>::template internal_pointer_type<T>;

  /**
   * Constructs an instance of \ref sycldnn::backend::OpenCLBackend from a
   * SYCL queue. Retains the underlying `cl_command_queue` which is
   * released on destruction.
   * \param queue The SYCL queue to construct the backend from.
   */
  OpenCLBackend(cl::sycl::queue& queue)
      : queue_{queue}, cl_queue_{queue.get()} {}

  /** Explicit destructor releases cl_queue_ */
  ~OpenCLBackend() {
    clReleaseCommandQueue(cl_queue_);
  }

  /**
   * Deleted copy constructor.
   */
  SNN_DISABLE_COPY(OpenCLBackend);
  /**
   * Deleted move constructor.
   */
  SNN_DISABLE_MOVE(OpenCLBackend);

  /**
   * Gets the SYCL queue that the backend is bound to.
   * \return Returns the SYCL queue that the backend is bound to.
   */
  cl::sycl::queue get_queue() { return queue_; }

  /**
   * Conversion function from external to internal pointer representation.
   * Is a no-op for clBLASBackend.
   * \param ptr Pointer to convert from
   * \return The passed-in paramter
   */
  template <typename T>
  internal_pointer_type<T> to_internal_pointer(pointer_type<T> ptr) {
    return ptr;
  }

  /**
   * Explicit release function for device memory. Is a no-op for this
   * backend.
   * \param ptr The pointer to deallocate
   */
  template <typename T>
  void release_internal_pointer(internal_pointer_type<T> ptr) {
    SNN_UNUSED_VAR(ptr);
  }

  /**
   * Get a MemObject containing the buffer corresponding to a given pointer.
   * \param ptr     A pointer referring to a SYCL buffer with some offset.
   * \param n_elems The number of elements required within the MemObject.
   * \return Returns a MemObject corresponding to the pointer.
   */
  template <typename T>
  auto get_mem_object(pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_mem_object(ptr.get_buffer(), 0ull, 0ull)) {
    return make_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /** \copydoc get_mem_object */
  template <typename T>
  auto get_mem_object_internal(internal_pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_mem_object(ptr.get_buffer(), 0ull, 0ull)) {
    return make_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /**
   * Allocation function that creates an internal_pointer representing
   * memory on the device associated with queue_.
   * \param n_bytes The number of bytes to allocate on device
   * \return Pointer to the allocation
   */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_bytes) {
    return internal_pointer_type<T>{n_bytes};
  }

  /**
   * Deallocate a device pointer.
   * \param ptr The pointer representing the buffer to deallocate.
   */
  template <typename T>
  void deallocate(internal_pointer_type<T> ptr) {
    SNN_UNUSED_VAR(ptr);
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_OPENCL_BACKEND_H_

