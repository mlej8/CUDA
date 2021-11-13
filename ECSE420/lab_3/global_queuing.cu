#include <iostream>
#include <unordered_set>

#include "io.hpp"
#include "logic_gates.cuh"
#include "read_input.hpp"

using namespace std;

__global__ void global_queuing(int numCurrLevelNodes,
                               int *currLevelNodes_h,
                               int *nodePtrs_h,
                               int *nodeNeighbors_h,
                               int *nodeVisited_h,
                               int *nodeOutput_h,
                               int *nodeGate_h,
                               int *nodeInput_h,
                               int *nextLevelNodes_h,
                               int *numNextLevelNodes_h) {
    int stride = blockDim.x * gridDim.x; // number of blocks * number of threads per block
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    //Loop over all nodes in the current level
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
                nodeOutput_h[neighbor] = logic_gate(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);

                // atomicAdd: reads a word at some address in global or shared memory, adds a number to it, and writes the result back to the same address. 
                // atomic means that it is guaranteed to be performed without interference from other threads. In other words, no other thread can access this address until the operation is complete.
                // add it to the global queue
                atomicExch(&nextLevelNodes_h[atomicAdd(numNextLevelNodes_h, 1)], neighbor);  // TODO find the index
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 9) {
        cout << "Usage: ./global_queuing <path_to_input_1.raw> <path_to_input_2.raw> "
                "<path_to_input_3.raw> <path_to_input_4.raw> "
                "<output_nodeOutput_filepath> <output_nextLevelNodes_filepath> <numBlock> <blockSize>"
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

    unordered_set<int> valid_block_size{32, 64, 128};
    unordered_set<int> valid_num_blocks{10, 25, 35};
    if (valid_num_blocks.find(numBlock) == valid_num_blocks.end() || valid_block_size.find(blockSize) == valid_block_size.end()) {
        cout << "The valid block sizes are:\n";
        for (auto bs : valid_block_size) {
            cout << bs << "\n";
        }

        cout << "The valid number of blocks are:\n";
        for (auto block_num : valid_num_blocks) {
            cout << block_num << "\n";
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
    cudaMemcpy(gpu_nodePtrs_h, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_nodeNeighbors_h, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_nodeVisited_h, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_nodeGate_h, nodeGate_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_nodeInput_h, nodeInput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_nodeOutput_h, nodeOutput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_currLevelNodes_h, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

    // using unified memory for global queue
    cudaMallocManaged(&nextLevelNodes_h, numTotalNeighbors_h * sizeof(int));
    cudaMallocManaged(&numNextLevelNodes_h, sizeof(int));

    // cout << *numNextLevelNodes_h;

    // launching kernel
    global_queuing<<<numBlock, blockSize>>>(
        numCurrLevelNodes,
        gpu_currLevelNodes_h,
        gpu_nodePtrs_h,
        gpu_nodeNeighbors_h,
        gpu_nodeVisited_h,
        gpu_nodeOutput_h,
        gpu_nodeGate_h,
        gpu_nodeInput_h,
        nextLevelNodes_h,
        numNextLevelNodes_h);
    cudaDeviceSynchronize();

    // copy output back to CPU
    cudaMemcpy(nodeOutput_h, gpu_nodeOutput_h, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    WriteOutput(output_nodeOutput_filepath, nodeOutput_h, numNodes);
    WriteOutput(output_nextLevelNodes_filepath, nextLevelNodes_h, *numNextLevelNodes_h);
    cudaFree(gpu_nodePtrs_h);
    cudaFree(gpu_nodeNeighbors_h);
    cudaFree(gpu_nodeVisited_h);
    cudaFree(gpu_currLevelNodes_h);
    cudaFree(gpu_nodeGate_h);
    cudaFree(gpu_nodeInput_h);
    cudaFree(gpu_nodeOutput_h);
    free(nodePtrs_h);
    free(nodeNeighbors_h);
    free(nodeVisited_h);
    free(currLevelNodes_h);
    free(nodeGate_h);
    free(nodeInput_h);
    free(nodeOutput_h);

    return 0;
}