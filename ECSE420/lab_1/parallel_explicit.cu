#include <iostream>

#include "io.hpp"
#include "logic_gates.cuh"

using namespace std;

int main(int argc, char *argv[]) {
  char *input_file_path, *output_file_path;
  int input_file_length;
  if (argc != 4) {
    cout << "./parallel_explicit <input_file_path> <input_file_length> "
            "<output_file_path>"
         << endl;
    exit(1);
  } else {
    input_file_path = argv[1];
    output_file_path = argv[3];
    input_file_length = stoi(argv[2]);
  }

  // 1. declare and allocate host and device memory
  size_t data_size = input_file_length * 3 * sizeof(uint8_t);
  size_t output_size = input_file_length * sizeof(uint8_t);
  char *data = new char[data_size];
  char *output = new char[output_size];
  char *gpu_data, *logic_gate_output;
  cudaMalloc(&gpu_data, data_size);
  cudaMalloc(&logic_gate_output, output_size);

  ReadCSV(input_file_path, data);

  // 2. copy/transfer data from host to device
  cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice);

  // assuming thhread size of 1024
  int num_blocks = 1, block_size = 1024;

  if (input_file_length > 1024) {
    num_blocks = (input_file_length + block_size - 1) / block_size;
  }
  // rounding up in case image size is not a multiple of block_size
  logic_gate<<<num_blocks, block_size>>>(gpu_data, logic_gate_output, input_file_length);
  cudaDeviceSynchronize();

  // copy/transfer data from device to host
  cudaMemcpy(output, logic_gate_output, output_size, cudaMemcpyDeviceToHost);

  // write final output
  WriteOutput(output_file_path, output, input_file_length);
  return 0;
}
