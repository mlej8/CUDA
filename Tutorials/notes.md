# CUDA Notes

## [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
The CUDA parallel programming model is designed to overcome this challenge of developing application software that transparently scales its parallelism to leverage the increasing number of processor cores while maintaining a low learning curve for programmers familiar with standard programming languages such as C. CUDA extends the C/C++ language to support parallel computing.

At its core are three key abstractions:
1. A hierarchy of thread groups
2. shared memories
3. barrier synchronization 

which are simply exposed to the programmer as a minimal set of language extensions.

These abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task parallelism. 
They guide the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer pieces that can be solved cooperatively in parallel by all threads within the block.

This decomposition preserves language expressivity by allowing threads to cooperate when solving each sub-problem, and at the same time enables automatic scalability. Indeed, each block of threads can be scheduled on any of the available multiprocessors within a GPU, in any order, concurrently or sequentially, so that a compiled CUDA program can execute on any number of multiprocessors as illustrated by Figure 3, and only the runtime system needs to know the physical multiprocessor count.

Note: A GPU is built around an array of Streaming Multiprocessors (SMs) (see Hardware Implementation for more details). 
A multithreaded program is partitioned into blocks of threads that execute independently from each other, so that a GPU with more multiprocessors will automatically execute the program in less time than a GPU with fewer multiprocessors.

## CUDA execution flow
1. Allocate host memory and initialize host data (input data)
2. Allocate device memory
3. Transfer input data from host to device memory.
4. Execute kernels
5. Transfer output (result) from device memory to host memory.

## Concurrent Programming vs Parallel Programming
Parallel: Programs are executed simultaneously on separate hardware, independent of each other.
Concurrent: Programs seem to run simultaneously on the same/separate hardware.

## CUDA Kernel Execution Configurations
CUDA kernel launches are specified using the triple angle bracket syntax: <<< >>>.

`kernel<<<num_thread_block, num_threads_per_block, shared_memory_size, stream>>>`

Functions on GPU called from CPU are declared using `__global__`
- Runs on the device
- Called from host code

Functions on GPU called from GPU are declared using `__device__`
- Runs on the device
- is called from device code

### Built-in CUDA variables
A kernel is launched as a grid of blocks of threads.

`threadIdx.x`: index of the current thread within its block

`blockIdx.x`: index of the current thread block within the grid

`blockDim.x`: number of threads in a block.

`gridDim.x`: number of blocks in the grid.

### CUDA Threads
`threadIdx` is a 3-component vector, so that threads can be identified using a one-dimensional, two-dimensional, or three-dimensional `thread` index forming a one-dimensional, two-dimensional, or three-dimensional block of threads, called a thread block.
**There is a limit to the number of threads per block**, since all threads of a block are expected to reside on the same processor core and must share the limited memory resources of that core. **On current GPUs, a thread block may contain up to 1024 threads.**
However, a kernel can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the number of threads per block times the number of blocks.


### CUDA Blocks
Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional grid of thread blocks as illustrated by the figure below.

![image](https://user-images.githubusercontent.com/43357040/149708528-1ca686fc-4864-4670-82cd-aa23d904cb22.png)

The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.

Each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional unique index accessible within the kernel through the built-in `blockIdx` variable. 

**A thread block size of 16x16 (256 threads)**, although arbitrary in this case, is a common choice. 

**Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. This independence requirement allows thread blocks to be scheduled in any order across any number of cores as illustrated by the figure below**, enabling programmers to write code that scales with the number of cores.

![image](https://user-images.githubusercontent.com/43357040/149708772-8021459c-246e-4884-b56e-636a587773e3.png)

**Threads within a block can cooperate by sharing data through some shared memory** and **can synchronize their execution to coordinate memory accesses.** More precisely, one can specify synchronization points in the kernel by calling the `__syncthreads()` intrinsic function; `__syncthreads()` acts as a barrier at which all threads in the block must wait before any is allowed to proceed. Shared Memory gives an example of using shared memory. In addition to `__syncthreads()`, the Cooperative Groups API provides a rich set of thread-synchronization primitives.

For efficient cooperation, the **shared memory is expected to be a low-latency memory near each processor core** (much like an L1 cache) and `__syncthreads()` is expected to be lightweight.


## [CUDA Memory Hierachy](https://www.youtube.com/watch?v=OSpy-HoR0ac)

In this section we will be discussion the [CUDA Memory Model](https://www.youtube.com/watch?v=HQejUtJtBlg) which has a one to one correspondance to its thread hierarchy shown below. 

Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. All threads have access to the same global memory.

There are also **two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces.** The global, constant, and texture memory spaces are optimized for different memory usages (see Device Memory Accesses). Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats (see Texture and Surface Memory).

The global, constant, and texture memory spaces are persistent across kernel launches by the same application.


![image](https://user-images.githubusercontent.com/43357040/149709015-4190c67d-9e6a-42bc-a56b-c9e306c90bc2.png)

Just like there are three levels in the thread hierarchy, there are also three levels to this main memory model. 


![image](https://user-images.githubusercontent.com/43357040/149810601-ddc4e7b3-dfe9-40cd-b55b-25defe6c7046.png)

Each one of the green squares on the lower right represent a single GPU core. These CUDA cores are grouped together into a Streaming Multiprocessors (SMs). The SMs are represented in the bottom image as yellow rectangles grouped together as sets of CUDA cores. Understanding how blocks are mapped onto SMs is fundamental during the design of kernels to gain optimal computational performance from the GPU.

- **Shared memory** and **Registers** are called **on-chip device memory**, because they are physically located on the GPU's SMs.
- Local and Global memory are called **off-chip device memory** since they are not located on the GPU itself.

Taking a look at the closer look at the physical layout on a GPU helps understanding why the DRAM is called off-chip memory. 

![image](https://user-images.githubusercontent.com/43357040/149810114-a4c7292c-77a1-4f5b-980e-20d7f7d4a32f.png)

The chip is at the center, the region surrounded by the blue line is where the DRAM is located.

![image](https://user-images.githubusercontent.com/43357040/149810193-25c7ce16-7304-4cd4-9ebb-653bcaecd57e.png)

We want to move frequently used data to the fastest memory available to us. 


![image](https://user-images.githubusercontent.com/43357040/149806998-f7b30ae2-d671-4bf4-9fdf-a63f5d0136e0.png)

### Local Memory and Registers

At the lowest level, each thread has its own private **local memory** and **registers**. Each thread has its own private local memory that can't be accessed by any other threads. When a thread has completed its execution, any local memory related to that thread is automatically destroyed. Threads also have private registers that have the same scope and lifetime as their local memory but has drastically different performance characteristics. 

![image](https://user-images.githubusercontent.com/43357040/149811229-52759e6e-53a0-4e2e-ae93-b1530adf0fbe.png)

Any variable declared inside of a kernel is stored on a register. However, if the size of the contents stored in this variable is too large to fit into the register space, then the content will spill over into the local memory. **Register spilling into local memory is very undesirable, because registers are the fastest form of memory in CUDA, but local memory is one of the slowest regions (off-chip device memory -> DRAM).** Should not place more information in local variables than can be stored in the register space.

### Shared Memory
In the next level up the hierarchy, we have the shared memory which corresponds to blocks in the thread hierarchy. **Each block has its own shared memory that is visible and accessible to all the threads within that block.** Shared memory is a small chunk of memory that sits on the SMs directly. When a block has completed its execution the content of its shared memory is automatically destroyed.

![image](https://user-images.githubusercontent.com/43357040/149811758-2abe1fb1-6a27-4ed0-b955-572e8629918f.png)

1. Shared memory is implemented **on-chip** which means that it is extremely fast.
2. It allows threads within a block to communicate with each other.

### Global Memory
At the top of the hierarchy is global memory. Global memory corresponds to all of the grids (kernel launches) in the entire program. **The contents of global memory are visible to all threads in the entire program. Any thread in the entire system can read and write from/to global memory at any time.** This means that threads from different kernels can all read from the global memory. The lifetime of data stored in global memory last the duration of the entire program or it can be manually destroyed using the `cudaFree()` function from host code.

![image](https://user-images.githubusercontent.com/43357040/149810455-13889103-cadc-47a7-8545-82ed302e2a2d.png)

Anytime we allocate device memory using `cudaMalloc()` this is the region where that allocation takes place. 

- **Disadvantage**: Since global memory is stored in off-chip DRAM, it exhibits very slow speed relative to other memory spaces. The use of global memory can't be avoided since we must transfer data from host to device using this memory space. The goal is to minimize global memory traffic since it is so slow.
- **Advantage**: The upside of global memory is that it is very large.

### Constant Memory

One portion of the memory model that doesn't have a corresponding level in the thread hierarchy is the **constant memory**. Since GPUs don't have a big cache, we can implement a very simple kind of cache using constant memory. Constant memory is very large as it is located in the device's DRAM. **All threads have access to it but it is read-only memory. We want to use this space for data that is accessed frequently and whose content doesn't change throughout kernel execution.**

**Although constant memory is implemented in hardware in off-chip DRAM, its content is aggressively cached into on-chip memory. Therefore, using constant memory can substantially reduce global memory traffic throughout the kernel execution.** 

![image](https://user-images.githubusercontent.com/43357040/149812000-803dab0f-7c57-44f3-9cc2-57831cee5587.png)

![image](https://user-images.githubusercontent.com/43357040/149812689-0d19e388-3521-4d10-8090-97a8290d64da.png)


## Heterogeneous Programming

The CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C++ program. This is the case, for example, when the kernels execute on a GPU and the rest of the C++ program executes on a CPU.

The CUDA programming model also assumes that both the host and the device maintain their own separate memory spaces in DRAM, referred to as host memory and device memory, respectively. 
Therefore, a program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime (described in Programming Interface). 
This includes device memory allocation and deallocation as well as data transfer between host and device memory.

![image](https://user-images.githubusercontent.com/43357040/149709852-d842b5fa-0575-47b9-8f9d-8fc8c0f78c04.png)

Note: Serial code executes on the host while parallel code executes on the device.

Unified Memory provides managed memory to bridge the host and device memory spaces. Managed memory is accessible from all CPUs and GPUs in the system as a single, coherent memory image with a common address space. This capability enables oversubscription of device memory and can greatly simplify the task of porting applications by eliminating the need to explicitly mirror data on host and device. See [Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd) for an introduction to Unified Memory.

## [CUDA GPUs Execution model](https://www.youtube.com/watch?v=usY0643pYs8)

This section discuss how the GPU hardware works and why do thread blocks exist?

- Why do we divide the problems into blocks?
- When do the blocks run?
- If you have a lot of thread blocks, in what order do these blocks run?
- Thread blocks are about letting groups of threads cooperate. How can threads cooperate?
    - with what limitation ?


![image](https://user-images.githubusercontent.com/43357040/149821456-4b130fda-86ee-47d7-8c95-141960ce2a62.png)

At a high level, CUDA GPUs are composed of many Streaming Multiprocessors, or SMs (blue squares). Each SM consists of multiple simple processors (red squares) that can run bunch of parallel threads.
Each SM can run multiple concurrent thread blocks.
Different GPUs have different number of SMs. 


When you have a CUDA program with a bunch of threads divided into thread blocks, **the important thing to understand that the GPU is responsible for allocating thread blocks to SMs.** As a programmer, we only have to worry about giving the GPU a bit pile of blocks and the GPU will take care of assigning them to run on the hardware. **All the SMs run in parallel and independently.**

![image](https://user-images.githubusercontent.com/43357040/149821932-b0772048-f3c7-40d5-9449-c955d3ba444a.png)

Thread blocks can execute in any order relative to each other, which allows for transparent scalability in parallel execution of CUDA kernels

## CUDA Warps (Chapter 4.7 Kirk and Hwu)

Once a block is assigned to a SM, it is further divided into 32-thread units called warps. The size of warps is implementation-specific. In fact, warps are not part of the CUDA specification. However, knowledge of warps can be helpful in understanding and optimizing the performance of CUDA applications on particular generations of CUDA devices. The size of warps is a property of a CUDA device, which is in the `dev_prop.warpSize` field of the device query variable (`dev_prop` in this case).
**The warp is the unit of thread scheduling in SMs. An SM is designed to execute all threads in a warp following the single instruction, multiple data (SIMD) model. That is, at any instant in time, one instruction is fetched and executed for all threads in the warp.**
Note that these threads will apply the same instruction to different portions of the data. As a result, all threads in a warp will always have the same execution timing.


**Each thread block is partitioned into warps. The execution of warps are implemented by an SIMD hardware.**
This implementation technique helps to reduce hardware manufacturing cost, lower runtime operation electricity cost, and enable some optimizations in servicing memory accesses. 
The size of a warp can easily vary from implementation to implementation. Up to this point in time, all CUDA devices have used similar warp configurations where each warp consists of 32 threads. 
Thread blocks are partitioned into warps based on thread indices.
**For a block of which the size is not a multiple of 32, the last warp will be padded with extra threads to fill up the 32 threads.**
For example, if a block has 48 threads, it will be partitioned into two warps, and its warp 1 will be padded with 16 extra threads. 
**Therefore, thread blocks should optimize to be a multiple of 32 as much as possible.**
For blocks that consist of multiple dimensions of threads, the dimensions will be projected into a linear order before partitioning into warps.
**The SIMD hardware executes all threads of a warp as a bundle. An instruction is run for all threads in the same warp.**