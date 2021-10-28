#include <iostream>

#include "lodepng.hpp"
#include "wm.h"

using namespace std;

/**
 * @brief Perform 2x2 max-pooling.
 *
 * @param gpu_image pointer to image on gpu memory.
 * @param new_image pointer to new image.
 * @param og_img_width width of original image.
 * @param og_img_height height of original image.
 * @param num_channels number of channels of original image.
 */
__global__ void convolve(unsigned char *gpu_image, unsigned char *new_image,
                         unsigned int og_img_width, unsigned int og_img_height,
                         unsigned int num_channels, float *filter) {
    // new image width
    int new_width = og_img_width - 2;

    // index in the new image
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    // new image coordinates
    int i = index / new_width;  // integer division (discards any fractional remains)
    int j = index % new_width;
    int z = blockIdx.y;

    // original image coordinates
    int og_i = i + 1;
    int og_j = j + 1;

    // flat index for new image
    // int flat_index = i * new_width * (num_channels - 1) + j * (num_channels - 1) + z;
    if (z == 3) {  // not manipulating alpha channel (directly copying over)
        new_image[i * new_width * num_channels + j * num_channels + z] = gpu_image[og_i * og_img_width * 4 + og_j * 4 + z];
    } else {
        int sum =
            filter[0] * gpu_image[(og_i - 1) * og_img_width * 4 + (og_j - 1) * 4 + z] +
            filter[1] * gpu_image[(og_i - 1) * og_img_width * 4 + og_j * 4 + z] +
            filter[2] * gpu_image[(og_i - 1) * og_img_width * 4 + (og_j + 1) * 4 + z] +
            filter[3] * gpu_image[og_i * og_img_width * 4 + (og_j - 1) * 4 + z] +
            filter[4] * gpu_image[og_i * og_img_width * 4 + og_j * 4 + z] +
            filter[5] * gpu_image[og_i * og_img_width * 4 + (og_j + 1) * 4 + z] +
            filter[6] * gpu_image[(og_i + 1) * og_img_width * 4 + (og_j - 1) * 4 + z] +
            filter[7] * gpu_image[(og_i + 1) * og_img_width * 4 + og_j * 4 + z] +
            filter[8] * gpu_image[(og_i + 1) * og_img_width * 4 + (og_j + 1) * 4 + z];

        // clipping output
        if (sum > 255) {
            sum = 255;
        } else if (sum < 0) {
            sum = 0;
        }

        new_image[i * new_width * num_channels + j * num_channels + z] = sum;
    }
}

int main(int argc, char *argv[]) {
    char *input_img_filename, *output_img_filename;
    int block_size;
    if (argc != 4) {
        cout << "./convolve <name of input png> <name of output png> "
                "<# threads>"
             << endl;
        exit(1);
    } else {
        input_img_filename = argv[1];
        output_img_filename = argv[2];
        block_size = stoi(argv[3]);
    }
    // 1. declare and allocate host and device memory
    unsigned char *image, *gpu_image, *new_image_gpu, *new_image_cpu;
    float *filter;
    unsigned int error, width, height, num_channels = 4;
    // 2. loading input image (initialize host data)
    error = lodepng_decode32_file(&image, &width, &height, input_img_filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }

    size_t new_image_size = (width - 2) * (height - 2) * num_channels * sizeof(unsigned char);
    size_t image_size = width * height * num_channels * sizeof(unsigned char);

    cudaMalloc(&gpu_image, image_size);
    cudaMalloc(&filter, 3 * 3 * sizeof(float));  // size of filter

    // 3. copy/transfer data from host to device
    cudaMemcpy(gpu_image, image,
               image_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(filter, w, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // convolved image will be of size height − 2 by width − 2
    cudaMalloc(&new_image_gpu, new_image_size);
    new_image_cpu = (unsigned char *) malloc(new_image_size);

    // rounding up in case image size is not a multiple of block_size
    dim3 num_blocks(((width - 2) * (height - 2) + block_size - 1) / block_size,
                    num_channels, 1);  // ignoring alpha channel

    // execute kernels
    convolve<<<num_blocks, block_size>>>(gpu_image, new_image_gpu, width, height,
                                         num_channels, filter);

    // tell CPU to wait until all threads in kernel are done execution before
    // accessing the results
    cudaDeviceSynchronize();

    // 5. Transfer results from device to host
    cudaMemcpy(new_image_cpu, new_image_gpu, new_image_size, cudaMemcpyDeviceToHost);

    // write pooled image
    error = lodepng_encode32_file(output_img_filename, new_image_cpu, (width - 2),
                                  (height - 2));
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }

    // clean memory
    cudaFree(gpu_image);
    cudaFree(new_image_gpu);
    cudaFree(filter);
    free(image);
    free(new_image_cpu);
}
