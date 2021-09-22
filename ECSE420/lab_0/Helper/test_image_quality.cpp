#include <stdio.h>

#include "image_equality.hpp"

int main() {
  char* input_filename_1 = "/home/mlej8/projects/CUDA/ECSE420/lab_0/Test Images/Test_1_pooled.png";
  char* input_filename_2 = "/home/mlej8/projects/CUDA/ECSE420/lab_0/Test Images/Test_1_pooled.png";

  // get mean squared error between image1 and image2
  float MSE = get_MSE(input_filename_1, input_filename_2);

  if (MSE < MAX_MSE) {
    printf("Images are equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
  } else {
    printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
  }

  return 0;
}
