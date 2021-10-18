#include <iostream>

#include "io.hpp"
#include "logic_gates.h"

using namespace std;

string sequential(const char *data, int input_file_length) {
  string result;
  for (int i = 0; i < input_file_length * 3; i += 3)  {
    int x = (int)data[i];
    int y = (int)data[i+1];
    int gate_type = (int)data[i+2];

    switch(gate_type) {
      case AND:
        result += x & y;
        break;
      case OR:
        result += x | y;
        break;
      case NAND:
        result += !(x & y);
        break;
      case NOR:
        result += !(x + y);
        break;
      case XOR:
        result += x ^ y;
        break;
      case XNOR:
        result += x == y;
        break;
      default:
        cerr << "Error: Input gate '" << gate_type << "' invalid" << endl;
        result += -1;
        break;
    }
  }
  return result;
}

int main(int argc, char *argv[]) {
  char *input_file_path, *output_file_path;
  int input_file_length;
  if (argc != 4) {
    cout << "./sequential <input_file_path> <input_file_length> "
            "<output_file_path>"
         << endl;
    exit(1);
  } else {
    input_file_path = argv[1];
    output_file_path = argv[3];
    input_file_length = atoi(argv[2]);
  }
  
  char *data = new char[input_file_length * 3 * sizeof(int)];
  ReadCSV(input_file_path, data);
  WriteOutput(output_file_path, sequential(data, input_file_length).c_str(), input_file_length);
  return 0;
}
