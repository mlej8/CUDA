set(compare_results compare_results)
# set(logic_gate logic_gate)

# libraries
add_library(io STATIC io.cpp)
add_library(logic_gate STATIC logic_gates.cu logic_gates.cuh logic_gates.h)

# executables
add_executable(${compare_results} compareResults.cpp)

