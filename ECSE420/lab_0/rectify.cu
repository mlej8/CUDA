#include <stdio.h>

#include <iostream>

#include "lodepng.hpp"

#define CENTER 127

/**
 * @brief Rectification produces an output image by repeating the following operation on each pixel of an input image.
 *        output[i][j] = input[i][j] if input[i][j] >= 0 else 0
 * 
 * @param image: image to be rectified.
 * @param num_channels: number of channels in the image to be rectified.
 */
__global__ void rectify(unsigned char* image, int num_channels) {
    int index = threadIdx.x * num_channels + blockDim.x * blockIdx.x * num_channels + blockIdx.y;
    if (image[index] < CENTER) image[index] = CENTER;
}

int main(int argc, char* argv[]) {
    // validate input parameters
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " <name of input png> <name of output png> <# threads>." << std::endl;
        exit(1);
    }

    char* input_img_filename = argv[1];
    char* output_img_filename = argv[2];
    int block_size = std::stoi(argv[3]);

    // 1. declare and allocate host and device memory
    unsigned char *image, *rectified_image;
    unsigned int error, width, height, num_channels = 4;

    // 2. loading input image (initialize host data)
    error = lodepng_decode32_file(&image, &width, &height, input_img_filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }

    cudaMalloc(&rectified_image, width * height * num_channels * sizeof(unsigned char));

    // 3. copy/transfer data from host to device
    cudaMemcpy(rectified_image, image, width * height * num_channels * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    // rounding up in case image size is not a multiple of block_size
    dim3 num_blocks((width * height + (block_size - 1)) / block_size, num_channels, 1);

    // execute kernels
    rectify<<<num_blocks, block_size>>>(rectified_image, num_channels);

    // tell CPU to wait until all threads in kernel are done execution before
    // accessing the resultsa
    cudaDeviceSynchronize();

    // 5. Transfer results from device to host
    cudaMemcpy(image, rectified_image, width * height * num_channels *
    sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // write rectified image
    error = lodepng_encode32_file(output_img_filename, image, width, height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }

    // clean memory
    cudaFree(rectified_image);
    free(image);
}
