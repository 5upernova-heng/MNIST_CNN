cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(MNIST_CNN)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_PREFIX_PATH /home/supernova/torch/libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(MNIST_CNN "src/main.cpp"
"src/dataset.cpp" "src/network.cpp"
"include/dataset.h" "include/network.h"
)
target_link_libraries(MNIST_CNN "${TORCH_LIBRARIES}" -lpthread -lm)
target_include_directories(MNIST_CNN PRIVATE include)
set_property(TARGET MNIST_CNN PROPERTY CXX_STANDARD 17)

