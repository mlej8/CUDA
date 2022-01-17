#include <iostream>
#include <unordered_set>

#include "io.hpp"
#include "logic_gates.cuh"
#include "read_input.hpp"

using namespace std;

__global__ void block_queuing(int numCurrLevelNodes, int blockQueueCapacity,
                              int *currLevelNodes_h, int *nodePtrs_h,
                              int *nodeNeighbors_h, int *nodeVisited_h,
                              int *nodeOutput_h, int *nodeGate_h,
                              int *nodeInput_h, int *nextLevelNodes_h,
                              int *numNextLevelNodes_h) {
  // initialize shared memory queue - extern indicates that __shared__ array
  // will be allocated dynamically at kernel launch time (shared memory size is
  // passed from the host)
  extern __shared__ int block_queue[];

  // counter used by threads within the same block
  __shared__ int blockQueueCounter;

  // use first thread in the block to initialize counter
  if (threadIdx.x == 0) {
    blockQueueCounter = 0;
  }
  __syncthreads();

  int stride =
      blockDim.x * gridDim.x;  // number of threads per block * number of blocks
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  // Loop over all nodes in the current level
  for (int i = index; i < numCurrLevelNodes; i += stride) {
    // extract element that is currently in the queue
    int node = currLevelNodes_h[i];

    // loop over all neighbors of the node
    for (int j = nodePtrs_h[node]; j < nodePtrs_h[node + 1]; j++) {
      int neighbor = nodeNeighbors_h[j];

      // if the neighbors hasn't been visited yet
      if (!nodeVisited_h[neighbor]) {
        // mark it as visited
        nodeVisited_h[neighbor] = 1;

        // update node output
        nodeOutput_h[neighbor] = logic_gate(
            nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);

        /**
         * atomicAdd: reads a word at some address in global or shared memory,
         * adds a number to it, and writes the result back to the same address.
         * atomic means that it is guaranteed to be performed without
         * interference from other threads. In other words, no other thread can
         * access this address until the operation is complete.
         */

        if (blockQueueCounter <
            blockQueueCapacity) {  // add it to the block queue if there's still
                                   // space
          atomicExch(&block_queue[atomicAdd(&blockQueueCounter, 1)], neighbor);
        } else {  // queue is full, add it directly to the global queue
          atomicExch(&nextLevelNodes_h[atomicAdd(numNextLevelNodes_h, 1)],
                     neighbor);
          // TODO optimize to write shared memory to global memory once full
          // TODO allocate space for block queue to go into global queue
          // (instead of at the end...) -> restore blockQueueCounter
          // TODO store block queue in global queue
          // atomicExch(&blockQueueCounter, 0);
          // cudaMemSet(block_queue, )
        }
      }
    }
  }
  // we need to synchronize threads here, because we want to wait for the
  // blockQueue to be updated by all threads in the block, before continuing
  // (copy data from shared queue of the current block to the global queue)
  // (resource:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#async_data_operations)
  __syncthreads();

  // add remaining data in the shared queue of the current block to the global
  // queue
  if (blockQueueCounter > 0) {
    for (int idx = threadIdx.x; idx < blockQueueCounter; idx += blockDim.x) {
      // copy data from shared memmory to global memory
      atomicExch(&nextLevelNodes_h[atomicAdd(numNextLevelNodes_h, 1)],
                 block_queue[idx]);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    cout
        << "Usage: ./block_queuing <path_to_input_1.raw> <path_to_input_2.raw> "
           "<path_to_input_3.raw> <path_to_input_4.raw> "
           "<output_nodeOutput_filepath> <output_nextLevelNodes_filepath> "
           "<numBlock> <blockSize> <sharedQueueSize>"
        << endl;
    exit(1);
  }
  char *input1 = argv[1];
  char *input2 = argv[2];
  char *input3 = argv[3];
  char *input4 = argv[4];
  char *output_nodeOutput_filepath = argv[5];
  char *output_nextLevelNodes_filepath = argv[6];
  int numBlock = atoi(argv[7]);
  int blockSize = atoi(argv[8]);
  int blockQueueCapacity = atoi(argv[9]);

  unordered_set<int> valid_block_size{32, 64};
  unordered_set<int> valid_num_blocks{25, 35};
  unordered_set<int> valid_block_queue_capacity{32, 64};
  if (valid_num_blocks.find(numBlock) == valid_num_blocks.end() ||
      valid_block_size.find(blockSize) == valid_block_size.end() ||
      valid_block_queue_capacity.find(blockQueueCapacity) ==
          valid_block_queue_capacity.end()) {
    cout << "The valid block sizes are:\n";
    for (auto bs : valid_block_size) {
      cout << bs << "\n";
    }

    cout << "The valid number of blocks are:\n";
    for (auto block_num : valid_num_blocks) {
      cout << block_num << "\n";
    }

    cout << "The valid number of block queue capacity are:\n";
    for (auto capacity : valid_block_queue_capacity) {
      cout << capacity << "\n";
    }
    exit(1);
  }

  // cpu variables
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int *nodeGate_h;
  int *nodeInput_h;
  int *nodeOutput_h;
  int *currLevelNodes_h;

  // read input files
  int numNodePtrs = read_input_one_two_four(&nodePtrs_h, input1);
  int numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, input2);
  int numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h,
                                  &nodeOutput_h, input3);
  int numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input4);

  // same variables on the gpu
  int *gpu_nodePtrs_h;
  int *gpu_nodeNeighbors_h;
  int *gpu_nodeVisited_h;
  int *gpu_nodeGate_h;
  int *gpu_nodeInput_h;
  int *gpu_nodeOutput_h;
  int *gpu_currLevelNodes_h;

  // global queue will be using unified memory
  int *nextLevelNodes_h;
  int *numNextLevelNodes_h;

  // allocate memory on the GPU
  cudaMalloc(&gpu_nodePtrs_h, numNodePtrs * sizeof(int));
  cudaMalloc(&gpu_nodeNeighbors_h, numTotalNeighbors_h * sizeof(int));
  cudaMalloc(&gpu_nodeVisited_h, numNodes * sizeof(int));
  cudaMalloc(&gpu_nodeGate_h, numNodes * sizeof(int));
  cudaMalloc(&gpu_nodeInput_h, numNodes * sizeof(int));
  cudaMalloc(&gpu_nodeOutput_h, numNodes * sizeof(int));
  cudaMalloc(&gpu_currLevelNodes_h, numCurrLevelNodes * sizeof(int));

  // copy content to GPU
  cudaMemcpy(gpu_nodePtrs_h, nodePtrs_h, numNodePtrs * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nodeNeighbors_h, nodeNeighbors_h,
             numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nodeVisited_h, nodeVisited_h, numNodes * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nodeGate_h, nodeGate_h, numNodes * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nodeInput_h, nodeInput_h, numNodes * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nodeOutput_h, nodeOutput_h, numNodes * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_currLevelNodes_h, currLevelNodes_h,
             numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

  // using unified memory for global queue
  cudaMallocManaged(&nextLevelNodes_h, numTotalNeighbors_h * sizeof(int));
  cudaMallocManaged(&numNextLevelNodes_h, sizeof(int));

  // launching kernel
  block_queuing<<<numBlock, blockSize,
                  blockQueueCapacity * sizeof(int)>>>(  // pass the size of the
                                                        // per-block shared
                                                        // queue (__shared__
                                                        // block_queue)
      numCurrLevelNodes, blockQueueCapacity, gpu_currLevelNodes_h,
      gpu_nodePtrs_h, gpu_nodeNeighbors_h, gpu_nodeVisited_h, gpu_nodeOutput_h,
      gpu_nodeGate_h, gpu_nodeInput_h, nextLevelNodes_h, numNextLevelNodes_h);
  cudaDeviceSynchronize();

  // copy output back to CPU
  cudaMemcpy(nodeOutput_h, gpu_nodeOutput_h, numNodes * sizeof(int),
             cudaMemcpyDeviceToHost);

  WriteOutput(output_nodeOutput_filepath, nodeOutput_h, numNodes);
  WriteOutput(output_nextLevelNodes_filepath, nextLevelNodes_h,
              *numNextLevelNodes_h);
  cudaFree(gpu_nodePtrs_h);
  cudaFree(gpu_nodeNeighbors_h);
  cudaFree(gpu_nodeVisited_h);
  cudaFree(gpu_currLevelNodes_h);
  cudaFree(gpu_nodeGate_h);
  cudaFree(gpu_nodeInput_h);
  cudaFree(gpu_nodeOutput_h);
  cudaFree(nextLevelNodes_h);
  cudaFree(numNextLevelNodes_h);
  free(nodePtrs_h);
  free(nodeNeighbors_h);
  free(nodeVisited_h);
  free(currLevelNodes_h);
  free(nodeGate_h);
  free(nodeInput_h);
  free(nodeOutput_h);

  return 0;
}
