// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
#include <iostream>

/**
 * @brief Function to add the elements of two arrays
 *        In order to turn the add function into a function that the GPU can run (called a kernel in CUDA)
 *        We need to add the specifier `__global__` to the function which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.
 *        Adding `__global__` transforms the function into a CUDA kernel function to add the elements of two arrays on the GPU
 *        These `__global__` functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.
 */
__global__ void
add(int n, float *x, float *y)
{
    /**
      * the following are variables that CUDA provides to let kernel get the indices of the running threads and thread blocks
      * threadIdx.x contains the index of the current thread within the block
      * blockDim.x contains the number of threads in the block
      * gridDim.x contains the number of blocks in the grid
      * blockIdx.x contains the index of the current thread block in the grid
      */

    // each thread gets its index by computing the offset to the beginning of its block + the thread's index within the
    // block (threadIdx.x)
    int index = blockDim.x * blockIdx.x + threadIdx.x; // idiomatic CUDA

    // in this case stride = total number of threads in the grid
    int stride = blockDim.x * gridDim.x;

    // this type of loop in a CUDA kernel is often called a `grid-stride loop`
    for (int i = index; i < n; i += stride) // NOTE: setting index to 0 and stride to 1 makes it semantically identical to the sequential version
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20; // 1M elements

    // allocate memory accessible by the GPU
    float *x, *y;

    /**
     *  UNIFIED memory https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
     *  To compute on the GPU, we need to allocate memory accessible by the GPU. 
     *  Unified Memory in CUDA makes this easy by providing a single space accessible by all GPUs and CPUs in your system
     *  To allocate data in unified memory, we call 
     *
     *          cudaMallocManaged()
     *
     *  which returns a pointer that you can access from the host (CPU) code or device (GPU) code.
     *  To free the data, just pass the pointer to cudaFree().
     * 
     *  Hence, we need to replace the calls to `new` in the code with calls to cudaMallocManaged(), and replace calls to delete [] with calls to cudaFree.
     */

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    /**
     * Launch the add() kernel, which invokes it on the GPU.
     * CUDA kernel launches are specified using the triple angle bracket syntax 
     * It represents the execution configuration and tells CUDA runtime how many parallel threads to use for the launch on the GPU
     * There are two parameters here:
     *  1. The first parameter of the execution configuration specifies the number of thread blocks.
     *  2. The second represents the number of threads in a thread block.
     * CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.
     * 
     *           <<<>>>
     * 
     * Using add<<<1, 256>>>(N, x, y);
     * __global__ void add(int n, float *x, float *y)
     *  {
     *      for (int i = 0; i < n; i++)
     *          y[i] = x[i] + y[i];
     *  }
     * 
     * will do the entire for loop computation once per thread, rather than spreading the computation across the parallel threads. 
     * We need to modify the kernel to spread the computation across parallel threads. 
     * CUDA C++ provides keywords that let kernels get the indices of the running threads. 
     * Specifically, threadIdx.x contains the index of the current thread within its block, and blockDim.x contains the number of threads in the block.
     * 
     * CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. 
     * Each SM can run multiple concurrent thread blocks.
     * As an example, a Tesla P100 GPU based on the Pascal GPU Architecture has 56 SMs, each capable of supporting up to 2048 active threads. 
     * To take full advantage of all these threads, I should launch the kernel with multiple thread blocks.
     * 
     * The blocks of parallel threads make up what is known as the `grid`.
     * Since I have N elements to process and 256 threads per block, I just need to calculate the number of blocks to get at least N threads.
     * I simply divide N by the block size (being careful to round up in case N is not a multiple of blockSize).
     * 
     * 
     */
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    add<<<num_blocks, block_size>>>(N, x, y);

    // we need to wait until the kernel is done before it accesses the results (because CUDA kernel launches don't block the calling CPU thread)
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}