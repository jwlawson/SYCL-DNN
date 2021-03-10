# Copyright Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.7)

include(SNNHelpers)
snn_include_guard(HANDLE_CLBLAS)

if(NOT SNN_DOWNLOAD_CLBLAS)
  find_package(clBLAS QUIET)
endif()

if(NOT clBLAS_FOUND AND (SNN_DOWNLOAD_CLBLAS OR SNN_DOWNLOAD_MISSING_DEPS))
  find_package(OpenCL REQUIRED)
  include(ExternalProject)
  set(clBLAS_REPO "https://github.com/clMathLibraries/clBLAS" CACHE STRING
    "clBLAS git repository to clone"
  )
  set(clBLAS_GIT_TAG "cf91139" CACHE STRING
    "Git tag, branch or commit to use for the clBLAS library"
  )
  set(clBLAS_DOWNLOAD_DIR ${sycldnn_BINARY_DIR}/clBLAS-src)
  set(clBLAS_SOURCE_DIR ${clBLAS_DOWNLOAD_DIR})
  set(clBLAS_BINARY_DIR ${sycldnn_BINARY_DIR}/clBLAS-build)
  set(clBLAS_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}clBLAS${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(clBLAS_LIBRARIES ${clBLAS_BINARY_DIR}/library/${clBLAS_LIBNAME})
  set(clBLAS_BYPRODUCTS ${clBLAS_LIBRARIES})
  if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    )
  endif()
  if(NOT TARGET clBLAS_download)
    ExternalProject_Add(clBLAS_download
      GIT_REPOSITORY    ${clBLAS_REPO}
      GIT_TAG           ${clBLAS_GIT_TAG}
      DOWNLOAD_DIR      ${clBLAS_DOWNLOAD_DIR}
      SOURCE_DIR        ${clBLAS_SOURCE_DIR}
      SOURCE_SUBDIR     src
      BINARY_DIR        ${clBLAS_BINARY_DIR}
      CMAKE_ARGS        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DBUILD_SHARED_LIBS=OFF
                        -DBUILD_TEST=OFF
                        -DBUILD_PERFORMANCE=OFF
                        -DBUILD_SAMPLE=OFF
                        -DBUILD_KTEST=OFF
                        -DBUILD_CLIENT=OFF
                        -DPRECOMPILE_GEMM_PRECISION_SGEMM=ON
                        -DPRECOMPILE_GEMM_TRANS_NN=ON
                        -DPRECOMPILE_GEMM_TRANS_TN=ON
                        -DPRECOMPILE_GEMM_TRANS_NT=ON
                        -DPRECOMPILE_GEMM_TRANS_TT=ON
                        -DAUTOGEMM_ARCHITECTURE=Fiji
                        ${cmake_toolchain}
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
      BUILD_BYPRODUCTS ${clBLAS_BYPRODUCTS}
    )
  endif()
  set(clBLAS_INCLUDE_DIR
    ${clBLAS_SOURCE_DIR}/src CACHE PATH
    "The clBLAS include directory" FORCE
  )
  set(clBLAS_INCLUDE_DIRS ${clBLAS_INCLUDE_DIR})
  # Have to explicitly make the include directories to add it to the library
  # target. This will be filled with the headers at build time when the
  # library is downloaded.
  file(MAKE_DIRECTORY ${clBLAS_INCLUDE_DIR})

  if(NOT TARGET clBLAS)
    add_library(clBLAS IMPORTED UNKNOWN)
    set_target_properties(clBLAS PROPERTIES
      IMPORTED_LOCATION ${clBLAS_LIBRARIES}
      INTERFACE_INCLUDE_DIRECTORIES ${clBLAS_INCLUDE_DIRS}
      INTERFACE_LINK_LIBRARIES OpenCL::OpenCL
    )
    add_dependencies(clBLAS clBLAS_download)
  endif()
  set(clBLAS_FOUND true)
  mark_as_advanced(clBLAS_REPO clBLAS_GIT_TAG clBLAS_INCLUDE_DIR)
endif()

if(NOT clBLAS_FOUND)
  message(FATAL_ERROR
    "Could not find clBLAS, consider setting SNN_DOWNLOAD_MISSING_DEPS")
endif()
