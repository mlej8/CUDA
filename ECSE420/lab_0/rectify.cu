#include <iostream>
#include <stdio.h>


__global__ void print_kernel() {
  printf("abc");
}

/**
 * Perform 2x2 max-pooling
 */
int main(int argc, char* argv[]) {
  print_kernel<<<1,1>>>();
  
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] <<" name of input png> <name of output png> <# threads>." << std::endl;
  }

  // split the image into 2x2 squares to determine number of threads needed
}
