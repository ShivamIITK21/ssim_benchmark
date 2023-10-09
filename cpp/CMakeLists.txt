cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ssim)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ssim ssim.cpp)
target_link_libraries(ssim "${TORCH_LIBRARIES}")
set_property(TARGET ssim PROPERTY CXX_STANDARD 17)