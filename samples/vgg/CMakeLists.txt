include(HandleSyclBLAS)
include(HandleCLBlast)
include(HandleCLBlas)

set(CMAKE_CXX_STANDARD 14)
snn_executable(
  WITH_SYCL
  TARGET
   vgg
  KERNEL_SOURCES
    vgg.cc
  PRIVATE_LIBRARIES
    sycl_dnn
    SyclBLAS::SyclBLAS
    clblast
    clBLAS
)
