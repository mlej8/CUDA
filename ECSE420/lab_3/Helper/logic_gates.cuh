#ifndef LAB_3_CUDA_LOGIC_GATES_H_
#define LAB_3_CUDA_LOGIC_GATES_H_

#include "logic_gates.h"

__global__ void logic_gate(const char *input, char *output, size_t input_file_length);
#endif
