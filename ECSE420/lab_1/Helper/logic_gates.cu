#include <stdio.h>
#include "logic_gates.cuh"

__global__ void logic_gate(const char *data, char *output, size_t input_file_length) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (index <  input_file_length) {
  int x = data[index * 3];
  int y = data[index * 3 + 1];
  int gate_type = data[index * 3 + 2];
  int result;
  switch (gate_type) {
    case AND:
      result = x & y;
      break;
    case OR:
      result = x | y;
      break;
    case NAND:
      result = !(x & y);
      break;
    case NOR:
      result = !(x + y);
      break;
    case XOR:
      result = x ^ y;
      break;
    case XNOR:
      result = x == y;
      break;
    default:
      printf("Error: Input gate '%d' invalid", gate_type);
      result = -1;
      break;
  }
  output[index] = result;
  }
}