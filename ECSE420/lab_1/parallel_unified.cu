#include <iostream>

#include "io.hpp"
#include "logic_gates.cuh"

using namespace std;

int main(int argc, char *argv[]) {
  char *input_file_path, *output_file_path;
  int input_file_length;
  if (argc != 4) {
    cout << "./parallel_unified <input_file_path> <input_file_length> <output_file_path>" << endl;
    exit(1);
  } else {
    input_file_path = argv[1];
    output_file_path = argv[3];
    input_file_length = atoi(argv[2]);
  }
  
  size_t data_size = input_file_length * 3 * sizeof(uint8_t);
  size_t output_size = input_file_length * sizeof(uint8_t);
  char *data, *output;
  cudaMallocManaged(&data, data_size);
  cudaMallocManaged(&output, output_size);
  ReadCSV(input_file_path, data);

  // prefetch memory onto GPU
  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(data, data_size, device, NULL);

  // assuming thhread size of 1024
  int num_blocks = 1, block_size = 1024;

  if (input_file_length > 1024) {
    num_blocks = (input_file_length + block_size - 1) / block_size;
  }
  // rounding up in case image size is not a multiple of block_size
  logic_gate<<<num_blocks, block_size>>>(data, output, input_file_length);
  cudaDeviceSynchronize();
  
  // write final output
  WriteOutput(output_file_path, output, input_file_length);
  return 0;
}
