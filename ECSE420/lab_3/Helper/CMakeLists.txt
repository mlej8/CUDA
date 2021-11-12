set(compare_results compare_results)
set(logic_gate logic_gate)
set(compare_next_level_nodes compare_next_level_nodes)
set(compare_node_output compare_node_output)
set(read_input read_input)
set(io io)

# libraries
add_library(${io} STATIC io.cpp io.hpp)
add_library(${read_input} STATIC read_input.cpp read_input.hpp)
add_library(${logic_gate} STATIC logic_gates.cu logic_gates.cuh logic_gates.h)

# executables
add_executable(${compare_results} compareResults.cpp)
add_executable(${compare_next_level_nodes} compareNextLevelNodes.cpp)
add_executable(${compare_node_output} compareNodeOutput.cpp)