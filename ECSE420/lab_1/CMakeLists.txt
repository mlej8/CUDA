# first command in a make file should always be specifying min cmake version required
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# required languages are (C++ and CUDA)
project(lab_0 LANGUAGES CXX CUDA VERSION 1.0) # lets CMake identify and verify the compiler it needs

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

# # binary file names
# set(sequential sequential)
# set(parallel_unified parallel_unified)
# set(parallel_explicit parallel_explicit)
# set(compare_results compare_results)
# set(logic_gate logic_gate)

# # create a list of libraries
# set(LIBS image_io image_equality)

# # include dirs
# set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/Helper")

# # add libraries
# add_subdirectory(Helper)

# # pooling
# add_executable(${pooling} pooling.cu)
# target_link_libraries(${pooling} PRIVATE ${LIBS})
# target_include_directories(${pooling} PUBLIC ${INCLUDE_DIR})

# # if debug build, add debugging flag for CUDA compiling 
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#   target_compile_options(${pooling} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
# endif()

# # rectifier
# add_executable(${rectify} rectify.cu)
# target_link_libraries(${rectify} PRIVATE ${LIBS})
# target_include_directories(${rectify} PUBLIC ${INCLUDE_DIR})

# # if debug build, add debugging flag for CUDA compiling 
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(${rectify} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
# endif()
