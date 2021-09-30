#include <iostream>

#include "lodepng.hpp"

using namespace std;

/**
 * @brief Perform 2x2 max-pooling.
 * 
 * @param gpu_image pointer to image on gpu memory.
 * @param new_image pointer to new image.
 * @param width width of image.
 * @param height height of image.
 * @param num_channels number of channels of the pooled image.
 */
__global__ void pool(unsigned char* gpu_image, unsigned char* new_image, unsigned int width,
                     unsigned int height, unsigned int num_channels) {
   
    // image index if it was 1D for a single channel
    int index = threadIdx.x * 2 + 2 * blockIdx.x * blockDim.x;
    int i = (index / width) * 2;
    int j = index % width;
    int z = blockIdx.y;
    int flat_index = i * width * 4 + j * 4 + z;
    unsigned char values[4] = {gpu_image[flat_index],                              // top left of 2x2 square
                               gpu_image[flat_index + num_channels],               // top right of 2x2 square
                               gpu_image[flat_index + width * 4],                  // bottom left of 2x2 square
                               gpu_image[flat_index + width * 4 + num_channels]};  // bottom right of 2x2 square
    unsigned char max_value = 0;
    for (int v = 0; v < 4; v++) {
        if (values[v] > max_value) {
            max_value = values[v];
        }
    }
    new_image[(i / 2) * (width / 2) * 4 + (j / 2) * 4 + z] = max_value;
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
    unsigned char *image, *gpu_image, *new_image;
    unsigned int error, width, height, num_channels = 4;

    // 2. loading input image (initialize host data)
    error = lodepng_decode32_file(&image, &width, &height, input_img_filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }

    cudaMalloc(&gpu_image, width * height * num_channels * sizeof(unsigned char));
    
    // pooled image is going to be twice as small on each dimension 
    // using unified memory - pooled image is accessible by both CPU and GPU
    cudaMallocManaged(&new_image,
                      (width / 2) * (height / 2) * num_channels * sizeof(unsigned char));

    // 3. copy/transfer data from host to device
    cudaMemcpy(gpu_image, image, width * height * num_channels * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    // rounding up in case image size is not a multiple of block_size
    dim3 num_blocks(((width / 2) * (height / 2) + block_size - 1) / block_size, num_channels, 1);

    // prefetch memory onto GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(new_image, (width / 2) * (height / 2) * num_channels * sizeof(unsigned char), device, NULL);

    // execute kernels
    pool<<<num_blocks, block_size>>>(gpu_image, new_image, width, height, num_channels);

    // tell CPU to wait until all threads in kernel are done execution before
    // accessing the resultsa
    cudaDeviceSynchronize();

    // 5. Transfer results from device to host
    // cudaMemcpy(image, gpu_image, width * height * num_channels *
    // sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // write pooled image
    error = lodepng_encode32_file(output_img_filename, new_image, (width / 2), (height / 2));
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }

    // clean memory
    cudaFree(gpu_image);
    cudaFree(new_image);
    free(image);
}
