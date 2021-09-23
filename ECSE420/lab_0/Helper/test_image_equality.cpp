#include <stdio.h>
#include <iostream>

#include "image_equality.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
    std::cout << "Usage: " << argv[0]
              << " <name of input png> <name of output png>."
              << std::endl;
  }

  char* input_filename_1 = argv[1];
  char* input_filename_2 = argv[2];

  // get mean squared error between image1 and image2
  float MSE = get_MSE(input_filename_1, input_filename_2);

  if (MSE < MAX_MSE) {
    printf("Images are equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
  } else {
    printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
  }

  return 0;
}
