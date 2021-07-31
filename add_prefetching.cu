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
     * UNIFIED MEMORY: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/ and https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
     * Unified Memory is a single memory address space accessible from any processor in a system.
     * To compute on the GPU, we need to allocate memory accessible by the GPU. 
     * Unified Memory in CUDA makes this easy by providing a single space accessible by all GPUs and CPUs in your system.
     * This hardware/software technology allows applications to allocate data that can be read or written from code running on either CPUs or GPUs.
     * Allocating Unified Memory is as simple as replacing calls to malloc() or `new` with calls to cudaMallocManaged()
     * To allocate data in unified memory, we call 
     *
     *          cudaMallocManaged()
     *
     * which returns a pointer that you can access from the host (CPU) code or device (GPU) code (any processor).
     * The memory pointed by this pointer is often called CUDA managed data. 
     * When code running on a CPU or GPU accesses data allocated this way, the CUDA system software and/or the hardware takes care of migrating memory pages to the memory of the accessing processor. 
     * The Pascal GPU architecture is the first with hardware support for ***virtual*** memory page faulting and migration, via its Page Migration Engine. 
     * Older GPUs based on the Kepler and Maxwell architectures also support a more limited form of Unified Memory.
     * 
     * To free the data, just pass the pointer to cudaFree().
     * 
     * Hence, we need to replace the calls to `new` in the code with calls to cudaMallocManaged(), and replace calls to delete [] with calls to cudaFree.
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
     * Prefetching
     * 
     * Use Unified Memory prefetching to move the data to the GPU after initializing it to eliminate the migration overhead (from host to device) and get a more accurate measurement of the vector add kernel performance.
     * CUDA provides cudaMemPrefetchAsync() for this purpose.
     * We can see that by doing this, there are no longer any GPU page faults reported.
     */
    // Prefetch the data to the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N * sizeof(float), device, NULL);
    cudaMemPrefetchAsync(y, N * sizeof(float), device, NULL);

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

/**
 * What Happens on Kepler When I call cudaMallocManaged()?
 * 
 * On systems with pre-Pascal GPUs like the NVIDIA GeForce 600 and 700 series, calling cudaMallocManaged() allocates `size` bytes of managed memory on the GPU device that is active when the call is made.
 * Internally, the driver also sets up page table entries in the GPU (device) memory for all pages covered by the allocation, so that the system knows that the pages are resident on that GPU.
 * 
 * So, in our example, running on a Tesla K80 GPU (Kepler architecture), x and y are both initially fully resident in GPU memory. 
 * Then in the loop starting on line 6, the CPU steps through both arrays, initializing their elements to 1.0f and 2.0f, respectively. 
 * Since the pages are initially resident in device memory, a page fault occurs on the CPU for each array page to which it writes, and the GPU driver migrates the page from device memory to CPU memory. 
 * After the loop, all pages of the two arrays are resident in CPU memory.
 * 
 * After initializing the data on the CPU, the program launches the add() kernel to add the elements of x to the elements of y: add<<<1, 256>>>(N, x, y);
 * 
 * On pre-Pascal GPUs, upon launching a kernel, the CUDA runtime must migrate all pages previously migrated to host memory or to another GPU back to the device memory of the device running the kernel. 
 * Since these older GPUs can’t page fault, all data must be resident on the GPU just in case the kernel accesses it (even if it won’t). 
 * This means there is potentially migration overhead on each kernel launch.
 */

/**
 * What Happens on Pascal When I call cudaMallocManaged()?
 * 
 * On Pascal and later GPUs, managed memory may not be physically allocated when cudaMallocManaged() returns; it may only be populated on access (or prefetching). 
 * In other words, pages and page table entries may not be created until they are accessed by the GPU or the CPU. 
 * The pages can migrate to any processor’s memory at any time, and the driver employs heuristics to maintain data locality and prevent excessive page faults. 
 * (Note: Applications can guide the driver using cudaMemAdvise(), and explicitly migrate memory using cudaMemPrefetchAsync(), as this blog post describes).
 * 
 * Unlike the pre-Pascal GPUs, the Tesla P100 supports hardware page faulting and migration. 
 * So in this case the runtime doesn’t automatically copy all the pages back to the GPU before running the kernel. 
 * The kernel launches without any migration overhead, and when it accesses any absent pages, the GPU stalls execution of the accessing threads, and the Page Migration Engine migrates the pages to the device before resuming the threads.
 * This means that the cost of the migrations is included in the kernel run time when I run my program on the Tesla P100 (2.1192 ms). 
 * 
 * In this kernel, every page in the arrays is written by the CPU, and then accessed by the CUDA kernel on the GPU, causing the kernel to wait on a lot of page migrations. 
 * That’s why the kernel time measured by the profiler is longer on a Pascal GPU like Tesla P100. 
 * Let’s look at the full nvprof output for the program on P100.
 * 
 * To eliminate/change the migration overhead to get a more accurate measurement of the vector add kernel performance:
 *  1. Move the data initialization to the GPU in another CUDA kernel.
 *  2. Run the kernel many times and look at the average and minimum run times. 
 *  3. Prefetch the data to GPU memory before running the kernel.
 * 
 */