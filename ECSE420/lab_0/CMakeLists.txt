# first command in a mkae file should always be specifying min cmake version required
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# required languages are (C++ and CUDA)
project(lab_0 LANGUAGES CXX CUDA VERSION 1.0) # lets CMake identify and verify the compiler it needs

# let CUDA files be compiled according to C++11
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# specify C++ standard 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# separately compile CUDA files 
set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)

# choose CUDA Architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

# binary file names
set(rectify rectify)
set(pooling pool)

# create a list of libraries
set(LIBS image_io image_equality)

# include dirs
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/Helper")

# add libraries
add_subdirectory(Helper)

# rectifier
add_executable(${pooling} pooling.cu)
target_link_libraries(${pooling} PRIVATE ${LIBS})
target_include_directories(${pooling} PUBLIC ${INCLUDE_DIR})

# pooling
add_executable(${rectify} rectify.cu)
target_link_libraries(${rectify} PRIVATE ${LIBS})
target_include_directories(${rectify} PUBLIC ${INCLUDE_DIR})
