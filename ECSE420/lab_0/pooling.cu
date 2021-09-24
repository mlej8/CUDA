#include <iostream>

#include "image_equality.hpp"
#include "lodepng.hpp"

using namespace std;

/**
 * @brief Perform 2x2 max-pooling.
 * // TODO generate description for arguments
 */
__global__ void pool(unsigned char* gpu_image, unsigned char* new_image, unsigned int width,
                     unsigned int height, unsigned int num_channels) {
    // TODO add check here if out of bound index then skip it.
    // TODO ask TA how to do the three dimensions with threads
    // TODO how do we do so that one thread is responsible for more than one block?
    // int index = threadIdx.x;  // index of current thread within block
    // int stride = blockDim.x;  // number of threads in a block
    // for (int i = (index * 2) % ; i < height; i += stride * 2) {
    for (int i = 0; i < height; i += 2) {
        // for (int j = (index * 2) % width; j < width; j += stride * 2) {
        for (int j = 0; j < width; j += 2) {
            for (int z = 0; z < num_channels; z++) {
                //  _________
                //  |   |   |
                //  |___|___|
                //  |   |   |
                //  |___|___|
                int flat_index = i * width * 4 + j * 4 + z;
                unsigned char values[4] = {gpu_image[flat_index],                               // top left
                                           gpu_image[flat_index + num_channels],                // top right
                                           gpu_image[flat_index + width * 4],                   // bottom left
                                           gpu_image[flat_index + width * 4 + num_channels]};   // bottom right
                unsigned char max_value = 0;
                for (int v = 0; v < 4; v++) {
                    if (values[v] > max_value) {
                        max_value = values[v];
                    }
                }
                
                new_image[(i / 2) * (width / 2) * 4 + (j / 2) * 4 + z] = max_value;
            }
        }
        // // one thread doing each of all three dimensions
        // int x = 0;
        // int y = 0;
        // int z = 0;
        // split the image into 2x2 squares to determine number of threads needed,
        // e.g. each thread will be responsible for one square
        // variables defined within device code do not need to be specified as device
        // variables because they are assumed to reside on the device.
    }
}

int main(int argc, char* argv[]) {
    // validate input parameters
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " <name of input png> <name of output png> <# threads>." << std::endl;
    }

    char* input_img_filename = argv[1];
    char* output_img_filename = argv[2];
    int num_threads = std::stoi(argv[3]);

    // 1. declare and allocate host and device memory
    unsigned char *image, *gpu_image, *new_image;
    unsigned int error, width, height, num_blocks, num_channels = 4;

    // 2. loading input image (initialize host data)
    error = lodepng_decode32_file(&image, &width, &height, input_img_filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(error);
    }

    cudaMalloc(&gpu_image, width * height * num_channels * sizeof(unsigned char));
    // new pooled image is going to be twice as small on each dimension - the
    // pooled image is accessible by both CPU and GPU
    cudaMallocManaged(&new_image,
                      (width / 2) * (height / 2) * num_channels * sizeof(unsigned char));

    // 3. copy/transfer data from host to device
    cudaMemcpy(gpu_image, image, width * height * num_channels * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    // TODO optimize with number of blocks by computing number of total threads
    // needed to complete
    // num_blocks = (width / 2 * height / 2 * num_channels) / num_threads;
    num_blocks = 1;
    // TODO: what to do if we have more threads than needed - only use one thread
    // per 2x2 square
    // TODO: how can this be improved by leveraging all threads ? can we have more
    // than one thread per 2x2 cube
    // TODO: do we preserve the `a` dimension in rgba ?

    // execute kernels
    pool<<<num_blocks, num_threads>>>(gpu_image, new_image, width, height, num_channels);

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
