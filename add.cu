#include <iostream>

/**
 * @brief Function to add the elements of two arrays
 *        In order to turn the add function into a function that the GPU can run (called a kernel in CUDA)
 *        We need to add the specifier `__global__` to the function which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.
 *        Adding `__global__` transforms the function into a CUDA kernel function to add the elements of two arrays on the GPU
 *        These `__global__` functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.
 */
__global__ void add(int n, float *x, float *y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    // NOTE: setting index to 0 and stride to 1 makes it semantically identical to the sequential version
    for (int i = index; i < n; i += stride)
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
     * There are two parameters here, the second is the number of threads in a thread block.
     * CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.
     * 
     *           <<<>>>
     * 
     * This launches one GPU thread to run add()
     * 
     * Using add<<<1, 256>>>(N, x, y);
     * __global__ void add(int n, float *x, float *y)
     *  {
     *      for (int i = 0; i < n; i++)
     *          y[i] = x[i] + y[i];
     *  }
     * 
     * will do the computation once per thread, rather than spreading the computation across the parallel threads. 
     * We need to modify the kernel to spread the computation across parallel threads. 
     * CUDA C++ provides keywords that let kernels get the indices of the running threads. 
     * Specifically, threadIdx.x contains the index of the current thread within its block, and blockDim.x contains the number of threads in the block.
     * 
     */
    add<<<1, 256>>>(N, x, y);

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