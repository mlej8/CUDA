# first command in a make file should always be specifying min cmake version required
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# required languages are (C++ and CUDA)
project(lab_1 LANGUAGES CXX CUDA VERSION 1.0) # lets CMake identify and verify the compiler it needs

# let CUDA files be compiled according to C++14
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# specify C++ standard 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# separately compile CUDA files 
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# choose CUDA Architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

# binary file names
set(sequential sequential)
set(parallel_unified parallel_unified)
set(parallel_explicit parallel_explicit)

# include dirs
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/Helper")

# add libraries
add_subdirectory(Helper)

# create a list of libraries
set(LIBS io)
set(CUDA_LIBS io logic_gate)

# sequential
add_executable(${sequential} sequential.cpp Helper/io.hpp Helper/logic_gates.h)
target_link_libraries(${sequential} PRIVATE ${LIBS})
target_include_directories(${sequential} PUBLIC ${INCLUDE_DIR})

# if debug build, add debugging flag for CUDA compiling 
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${sequential} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()

add_executable(${parallel_unified} parallel_unified.cu Helper/io.hpp Helper/logic_gates.cuh)
target_link_libraries(${parallel_unified} PRIVATE ${CUDA_LIBS})
target_include_directories(${parallel_unified} PUBLIC ${INCLUDE_DIR})

# if debug build, add debugging flag for CUDA compiling 
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${parallel_unified} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()

add_executable(${parallel_explicit} parallel_explicit.cu Helper/io.hpp Helper/logic_gates.cuh)
target_link_libraries(${parallel_explicit} PRIVATE ${CUDA_LIBS})
target_include_directories(${parallel_explicit} PUBLIC ${INCLUDE_DIR})

# if debug build, add debugging flag for CUDA compiling 
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${parallel_explicit} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()