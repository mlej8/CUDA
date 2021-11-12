#include <iostream>
#include <queue>

#include "io.hpp"
#include "logic_gates.h"
#include "read_input.hpp"

using namespace std;

// string gate_solver(const char *data, int input_file_length) {
int gate_solver(int gate_type, int x, int y) {
  int result;
  // for (int i = 0; i < input_file_length * 3; i += 3) {
  //   int x = (int)data[i];
  //   int y = (int)data[i + 1];
  //   int gate_type = (int)data[i + 2];

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
        cerr << "Error: Input gate '" << gate_type << "' invalid" << endl;
        result = -1;
        break;
    }
  return result;
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    cout << "./sequential <path_to_input_1.raw> <path_to_input_2.raw> "
            "<path_to_input_3.raw> <path_to_input_4.raw> "
            "<output_nodeOutput_filepath> <output_nextLevelNodes_filepath>"
         << endl;
    exit(1);
  }
  char *input1 = argv[1];
  char *input2 = argv[2];
  char *input3 = argv[3];
  char *input4 = argv[4];
  char *output_nodeOutput_filepath = argv[5];
  char *output_nextLevelNodes_filepath = argv[6];

  // Variables
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int *currLevelNodes_h;
  int numNextLevelNodes_h;
  int *nodeGate_h;
  int *nodeInput_h;
  int *nodeOutput_h;

  // output
  int *nextLevelNodes_h;

  // read input files
  int numNodePtrs = read_input_one_two_four(&nodePtrs_h, input1);
  int numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, input2);
  int numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h,
                                  &nodeOutput_h, input3);
  int numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input4);

  queue<int> q;

  // loop over all ndoes in the current level
  for (int i = 0; i < numCurrLevelNodes; i++) {
    int node = currLevelNodes_h[i];

    // loop over all neighbors of the node
    for (int j = nodePtrs_h[node]; j < nodePtrs_h[node + 1]; j++) {
      int neighbor = nodeNeighbors_h[j];

      // if the neighbors hasn't been visited yet
      if (!nodeVisited_h[neighbor]) {
        // mark it as visited
        nodeVisited_h[neighbor] = 1;
        q.push(neighbor);

        nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);

        nextLevelNodes_h[numNextLevelNodes_h] = neighbor;
        ++(numNextLevelNodes_h);

        // add it to the queue
      }
    }
  }

    // Close and free pointers
  WriteOutput(output_nodeOutput_filepath, "abc", 2);
  WriteOutput(output_nextLevelNodes_filepath, "abc", 2);
  // fclose(output1);
  // fclose(output2);
  free(nextLevelNodes_h);
  free(nodePtrs_h);
  free(nodeNeighbors_h);
  free(nodeVisited_h);
  free(currLevelNodes_h);
  free(nodeGate_h);
  free(nodeInput_h);
  free(nodeOutput_h);

  return 0;
}