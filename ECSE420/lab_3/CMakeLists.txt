# first command in a make file should always be specifying min cmake version required
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# required languages are (C, C++ and CUDA)
project(lab_3 LANGUAGES C CXX CUDA VERSION 1.0) # lets CMake identify and verify the compiler it needs

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
set(global_queuing global_queuing)
set(block_queuing block_queuing)
set(compare_results compare_results)
set(logic_gate logic_gate)
set(compare_next_level_nodes compare_next_level_nodes)
set(compare_node_output compare_node_output)
set(read_input read_input)
set(io io)

# include dirs
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/Helper")

# add libraries
add_subdirectory(Helper)

# create a list of libraries
set(LIBS ${io} ${read_input})
set(CUDA_LIBS ${io} ${logic_gate} ${read_input})

add_executable(${sequential} sequential.cpp Helper/read_input.hpp Helper/io.hpp Helper/logic_gates.h)
target_link_libraries(${sequential} ${LIBS})
target_include_directories(${sequential} PUBLIC ${INCLUDE_DIR})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${sequential} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
endif ()

add_executable(${global_queuing} global_queuing.cu Helper/read_input.hpp)
target_link_libraries(${global_queuing} ${CUDA_LIBS})
target_include_directories(${global_queuing} PUBLIC ${INCLUDE_DIR})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${global_queuing} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif ()

add_executable(${block_queuing} block_queuing.cu Helper/read_input.hpp)
target_link_libraries(${block_queuing} ${CUDA_LIBS})
target_include_directories(${block_queuing} PUBLIC ${INCLUDE_DIR})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${block_queuing} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif ()