# [Unified Memory Tutorial](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
Unified Memory is a single memory address space accessible from any processor in a system.
This hardware/software technology allows applications to allocate data that can be read or written from code running on either CPUs or GPUs.

When code running on a CPU or GPU accesses data allocated this way (often called CUDA managed data), the CUDA system software and/or the hardware takes care of migrating memory pages to the memory of the accessing processor. 
**The important point here is that the Pascal GPU architecture is the first with hardware support for virtual memory page faulting and migration, via its Page Migration Engine**. 

Pascal GPUs such as the NVIDIA Titan X and the NVIDIA Tesla P100 are the first GPUs to include the Page Migration Engine.
**The Page Migration Engine is hardware support for Unified Memory page faulting and migration.**

## What Happens on Kepler When I call `cudaMallocManaged()`?
On systems with pre-Pascal GPUs like the Tesla K80, calling `cudaMallocManaged()` allocates `size` bytes of managed memory on the GPU device that is active when the call is made.
Internally, the driver also sets up page table entries for all pages covered by the allocation, so that the system knows that the pages are resident on that GPU.
In our example, running on a Tesla K80 GPU (Kepler architecture), `x` and `y` are both initially fully resident in GPU memory. 
Then in the loop initializing `x` and `y`, the CPU steps through both arrays, initializing their elements to `1.0f` and `2.0f`, respectively.
Since the pages are initially resident in device memory, a page fault occurs on the CPU for each array page to which it writes, and the GPU driver migrates the page from device memory to CPU memory. 
After the loop, all pages of the two arrays are resident in CPU memory.
After initializing the data on the CPU, the program launches the `add()` kernel to add the elements of `x` to the elements of `y`.

On pre-Pascal GPUs, upon launching a kernel, the CUDA runtime must migrate all pages previously migrated to host memory or to another GPU back to the device memory of the device running the kernel.
Since these older GPUs can’t page fault, all data must be resident on the GPU just in case the kernel accesses it (even if it won't).
This means there is potentially migration overhead on each kernel launch.


## What Happens on Pascal When I call `cudaMallocManaged()`?
On Pascal and later GPUs, managed memory may not be physically allocated when `cudaMallocManaged()` returns; it may only be populated on access (or prefetching). In other words, **pages and page table entries may not be created until they are accessed by the GPU or the CPU**. The pages can migrate to any processor's memory at any time, and the driver employs heuristics to maintain data locality and prevent excessive page faults.

Unlike the pre-Pascal GPUs, the `Tesla P100` supports hardware page faulting and migration. So in this case the **runtime doesn't automatically copy all the pages back to the GPU before running the kernel**. The **kernel launches without any migration overhead, and when it accesses any absent pages, the GPU stalls execution of the accessing threads, and the Page Migration Engine migrates the pages to the device before resuming the threads**.

**This means that the cost of the migrations is included in the kernel runtime** when I run my program on the `Tesla P100` (2.1192 ms). In this kernel, every page in the arrays is written by the CPU, and then accessed by the CUDA kernel on the GPU, causing the kernel to wait on a lot of page migrations. That’s why the kernel time measured by the profiler is longer on a Pascal GPU like `Tesla P100`.

## What Should I Do About This?
In a real application, the GPU is likely to perform a lot more computation on data (perhaps many times) without the CPU touching it. The migration overhead in this simple code is caused by the fact that the CPU initializes the data and the GPU only uses it once. There are a few different ways that I can eliminate or change the migration overhead to get a more accurate measurement of the vector add kernel performance.

1. Move the data initialization to the GPU in another CUDA kernel.
2. Run the kernel many times and look at the average and minimum run times.
3. Prefetch the data to GPU memory before running the kernel.

## A Note on Concurrency
Keep in mind that your system has multiple processors running parts of your CUDA application concurrently: one or more CPUs and one or more GPUs.
Even in our simple example, there is a CPU thread and one GPU execution context. 
Therefore, **we have to be careful when accessing the managed allocations on either processor, to ensure there are no race conditions (e.g. CPU and GPU access same memory at same time)**.

Simultaneous access to managed memory from the CPU and GPUs of compute capability lower than 6.0 is not possible. This is because pre-Pascal GPUs lack hardware page faulting, so coherence can’t be guaranteed. On these GPUs, an access from the CPU while a kernel is running will cause a segmentation fault.

On Pascal and later GPUs, the CPU and the GPU can simultaneously access managed memory, since they can both handle page faults; however, it is **up to the application developer to ensure there are no race conditions caused by simultaneous accesses.**

In our simple example, we have a call to `cudaDeviceSynchronize()` after the kernel launch. 
This **ensures that the kernel runs to completion before the CPU tries to read the results from the managed memory pointer. Otherwise, the CPU may read invalid data (on Pascal and later), or get a segmentation fault (on pre-Pascal GPUs).**

## The Benefits of Unified Memory on Pascal and Later GPUs
Starting with the Pascal GPU architecture, Unified Memory functionality is significantly improved with 49-bit virtual addressing and on-demand page migration. 
49-bit virtual addresses are sufficient to enable GPUs to access the entire system memory plus the memory of all GPUs in the system. 
The Page Migration engine allows GPU threads to fault on non-resident memory accesses so the system can migrate pages on demand from anywhere in the system to the GPU’s memory for efficient processing.

In other words, Unified Memory transparently enables oversubscribing GPU memory, enabling out-of-core computations for any code that is using Unified Memory for allocations (e.g. `cudaMallocManaged()`). 
It "just works" without any modifications to the application, whether running on one GPU or multiple GPUs.

Also, Pascal and Volta GPUs support system-wide atomic memory operations. 
That means you can atomically operate on values anywhere in the system from multiple GPUs. 
This is useful in writing efficient multi-GPU cooperative algorithms.

Demand paging can be particularly beneficial to applications that access data with a sparse pattern. 
In some applications, it’s not known ahead of time which specific memory addresses a particular processor will access. 
Without hardware page faulting, applications can only pre-load whole arrays, or suffer the cost of high-latency off-device accesses (also known as "Zero Copy”). 
But page faulting means that only the pages the kernel accesses need to be migrated.