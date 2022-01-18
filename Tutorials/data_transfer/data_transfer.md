# [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

The peak bandwidth between the device memory and the GPU is much higher (144 GB/s on the NVIDIA Tesla C2050, for example) than the peak bandwidth between host memory and device memory (8 GB/s on PCIe x16 Gen2). 
This disparity means that your **implementation of data transfers between the host and GPU devices can make or break your overall application performance.**
Transfers between the host and device are the slowest link of data movement involved in GPU computing, so you should take care to minimize transfers. 
Following the guidelines in this post can help you make sure necessary transfers are efficient. 
When you are porting or writing new CUDA C/C++ code, I recommend that you start with pageable transfers from existing host pointers. 
As I mentioned earlier, as you write more device code you will eliminate some of the intermediate transfers, so any effort you spend optimizing transfers early in porting may be wasted. 
Also, rather than instrument code with CUDA events or other timers to measure time spent for each transfer, use `nsys`, the command-line CUDA profiler.

Let's start with a few general guidelines for host-device data transfers:
1. **Minimize the amount of data transferred between host and device** as much as possible.
2. **Pinned Host Memory**: Higher bandwidth is possible between the host and the device when using page-locked (or "pinned") memory.
3. **Batching many small transfers into one larger transfer** performs much better because it eliminates most of the per-transfer overhead.
4. **Overlap data transfers**: Data transfers between the host and device can sometimes be overlapped with kernel execution and other data transfers.

## Minimizing Data Transfers
We should not use only the GPU execution time of a kernel relative to the execution time of its CPU implementation to decide whether to run the GPU or CPU version. 
We also need to consider the cost of moving data across the PCI-e bus, especially when we are initially porting code to CUDA. 
Because CUDA's heterogeneous programming model uses both the CPU and GPU, code can be ported to CUDA one kernel at a time. 
In the initial stages of porting, data transfers may dominate the overall execution time. 
It's worthwhile to keep tabs on time spent on data transfers separately from time spent in kernel execution. 
It's easy to use the `nsys` profiler for this.
As we port more of our code, we'll remove intermediate transfers and decrease the overall execution time correspondingly.
The reasoning is as we port more code to `device`, everything can be done on `device`, hence device to host transfer will be least frequent.

## [Pinned Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
Host (CPU) data allocations are pageable by default. 
The GPU cannot access data directly from pageable host memory, so when a data transfer from pageable host memory to device memory is invoked, the CUDA driver must first allocate a temporary page-locked, or "pinned" host array, copy the host data to the pinned array, and then transfer the data from the pinned array to device memory, as illustrated below:
![pinned memory](https://user-images.githubusercontent.com/43357040/149685248-fa14bde7-68ab-4bee-b4cd-c1a9b3962d9c.png) 

Pinned memory is used as a staging area for transfers from the device to the host. 
We can avoid the cost of the transfer between pageable and pinned host arrays by directly allocating our host arrays in pinned memory. 
Allocate pinned host memory in CUDA C/C++ using `cudaMallocHost()` or `cudaHostAlloc()`, and deallocate it with `cudaFreeHost()`. 

In every example until this point, we have used the `cudaMalloc()` function to allocate memory on the host, which **allocates standard pageable memory on the host**. CUDA provides another API called `cudaHostAlloc()`, which allocates page-locked host memory or what is sometimes referred to as pinned memory. It **guarantees that the operating system will never page this memory out to the disk and that it will remain in physical memory. So, any application can access the physical address of the buffer This property helps the GPU copy data to and from the host via Direct Memory Access (DMA) without CPU intervention. This helps improve the performance of memory transfer operations.**

The GPU always must DMA from pinned memory. If you use `cudaMalloc()` for your host data, then it is in pageable (non-pinned memory). **When you call `cudaMemcpy()`, the CUDA driver has to first memcpy the data from your non-pinned pointer to an internal pinned memory pointer, and then the host->GPU DMA can be invoked.**

If you allocate your host memory with `cudaMallocHost()` and initialize the data there directly, then the **driver doesn't have to memcpy from pageable to pinned memory before DMAing – it can DMA directly.** 
That is why it is faster.

It is possible for pinned memory allocation to fail, so you should always check for errors. 
The following code excerpt demonstrates allocation of pinned memory with error checking.
```
cudaError_t status = cudaMallocHost((void**)&h_aPinned, bytes);
if (status != cudaSuccess)
  printf("Error allocating pinned host memory\n");
```

Data transfers using host pinned memory use the same `cudaMemcpy()` syntax as transfers with pageable memory. We can use `bandwidth_test.cu` in the current directory to compare pageable and pinned transfer rates.

**You should not over-allocate pinned memory. 
Doing so can reduce overall system performance because it reduces the amount of physical memory available to the operating system and other programs.** 
How much is too much is difficult to tell in advance, so as with all optimizations, test your applications and the systems they run on for optimal performance parameters.


## Batching Small Transfers
Due to the overhead associated with each transfer, it is preferable to batch many small transfers together into a single transfer.
This is easy to do by using a temporary array, preferably pinned, and packing it with the data to be transferred.

For two-dimensional array transfers, you can use `cudaMemcpy2D()`.
```
cudaMemcpy2D(dest, dest_pitch, src, src_pitch, w, h, cudaMemcpyHostToDevice)
```
The arguments here are a pointer to the first destination element and the pitch of the destination array, a pointer to the first source element and pitch of the source array, the width and height of the submatrix to transfer, and the memcpy kind.
There is also a `cudaMemcpy3D()` function for transfers of rank three array sections.


## [Overlap Data Transfers](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
We can overlap data transfers with computation on the host, computation on the device, and in some cases other data transfers between the host and device. Achieving overlap between data transfers and other operations requires the use of CUDA streams, so first let's learn about streams.

### CUDA Streams
A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued by the host code.
While operations within a stream are guaranteed to execute in the prescribed order, operations in different streams can be interleaved and, when possible, they can even run concurrently.

#### The default stream
All device operations (kernels and data transfers) in CUDA run in a stream. 
When no stream is specified, the default stream (also called the "null stream") is used. 
The default stream is different from other streams because it is a synchronizing stream with respect to operations on the device: no operation in the default stream will begin until all previously issued operations in any stream on the device have completed, and an operation in the default stream must complete before any other operation (in any stream on the device) will begin.

Please note that CUDA 7, released in 2015, introduced a new option to use a separate default stream per host thread, and to treat per-thread default streams as regular streams (i.e. they don't synchronize with operations in other streams). Read more about this new behavior in the post [GPU Pro Tip: CUDA 7 Streams Simplify Concurrency](https://developer.nvidia.com/blog/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/).

Let's look at some simple code examples that use the default stream, and discuss how operations progress from the perspective of the host as well as the device.

```
cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
increment<<<1,N>>>(d_a)
cudaMemcpy(a, d_a, numBytes, cudaMemcpyDeviceToHost);
```

In the code above, from the perspective of the device, all three operations are issued to the same (default) stream and will execute in the order that they were issued.

From the perspective of the host, the implicit data transfers are blocking or synchronous transfers, while the kernel launch is asynchronous. 
Since the host-to-device data transfer on the first line is synchronous, the CPU thread will not reach the kernel call on the second line until the host-to-device transfer is complete. 
Once the kernel is issued, the CPU thread moves to the third line, but the transfer on that line cannot begin due to the device-side order of execution.

The asynchronous behavior of kernel launches from the host's perspective makes overlapping device and host computation very simple. 
We can modify the code to add some independent CPU computation as follows.

```
cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
increment<<<1,N>>>(d_a)
myCpuFunction(b)
cudaMemcpy(a, d_a, numBytes, cudaMemcpyDeviceToHost);
```

In the above code, as soon as the `increment()` kernel is launched on the device the CPU thread executes `myCpuFunction()`, overlapping its execution on the CPU with the kernel execution on the GPU. 
Whether the host function or device kernel completes first doesn't affect the subsequent device-to-host transfer, which will begin only after the kernel completes.  
From the perspective of the device, nothing has changed from the previous example; the device is completely unaware of `myCpuFunction()`.


#### Non-default streams
Non-default streams in CUDA C/C++ are declared, created, and destroyed in host code as follows.

```
cudaStream_t stream1;
cudaError_t result;
result = cudaStreamCreate(&stream1)
result = cudaStreamDestroy(stream1)
```

To issue a data transfer to a non-default stream we use the `cudaMemcpyAsync()` function, which is similar to the `cudaMemcpy()` function discussed in the previous post, but takes a stream identifier as a fifth argument.

```
result = cudaMemcpyAsync(d_a, a, N, cudaMemcpyHostToDevice, stream1)
```

`cudaMemcpyAsync()` is non-blocking on the host, so control returns to the host thread immediately after the transfer is issued. There are `cudaMemcpy2DAsync()` and `cudaMemcpy3DAsync()` variants of this routine which can transfer 2D and 3D array sections asynchronously in the specified streams.

**To issue a kernel to a non-default stream we specify the stream identifier as a fourth execution configuration parameter** (the third execution configuration parameter allocates shared device memory, which we'll talk about later; use 0 for now).

```
increment<<<1,N,0,stream1>>>(d_a)
```

#### Synchronization with streams
Since all operations in non-default streams are non-blocking with respect to the host code, you will run across situations where you need to synchronize the host code with operations in a stream.
There are several ways to do this. 
The "heavy hammer" way is to use `cudaDeviceSynchronize()`, **which blocks the host code until all previously issued operations on the device have completed.** In most cases this is overkill, and can really hurt performance due to stalling the entire device and host thread.

The `CUDA stream API` has multiple less severe methods of synchronizing the host with a stream.  
The function `cudaStreamSynchronize(stream)` can be used to block the host thread until all previously issued operations in the specified stream have completed. 
The function `cudaStreamQuery(stream)` tests whether all operations issued to the specified stream have completed, without blocking host execution.
The functions `cudaEventSynchronize(event)` and `cudaEventQuery(event)` act similar to their stream counterparts, except that their result is based on whether a specified event has been recorded rather than whether a specified stream is idle. 
You can also synchronize operations within a single stream on a specific event using `cudaStreamWaitEvent(event)` (even if the event is recorded in a different stream, or on a different device!).



#### Overlapping Kernel Execution and Data Transfers
Earlier we demonstrated how to overlap kernel execution in the default stream with execution of code on the host. 
But our main goal in this post is to show you how to overlap kernel execution with data transfers. 
There are several requirements for this to happen.

- The device must be capable of "concurrent copy and execution".
This can be queried from the `deviceOverlap` field of a `cudaDeviceProp` struct, or from the output of the deviceQuery sample included with the CUDA SDK/Toolkit. Nearly all devices with compute capability 1.1 and higher have this capability.
- **The kernel execution and the data transfer to be overlapped must both occur in different, non-default streams.**
- **The host memory involved in the data transfer must be pinned memory.**

So let's modify our simple host code from above to use multiple streams and see if we can achieve any overlap. 
In the modified code, we **break up the array of size N into chunks of `streamSize` elements. Since the kernel operates independently on all elements, each of the chunks can be processed independently. The number of (non-default) streams used is `nStreams=N/streamSize`.**
There are multiple ways to implement the domain decomposition of the data and processing; one is to loop over all the operations for each chunk of the array as in this example code.


```
for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
  kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
}
```

Another approach is to batch similar operations together, issuing all the host-to-device transfers first, followed by all kernel launches, and then all device-to-host transfers, as in the following code.

```
for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&d_a[offset], &a[offset], 
                  streamBytes, cudaMemcpyHostToDevice, cudaMemcpyHostToDevice, stream[i]);
}

for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
}

for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&a[offset], &d_a[offset], 
                  streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[i]);
}
```


Both asynchronous methods shown above yield correct results, and in both cases **dependent operations are issued to the same stream in the order in which they need to be executed**. But the two approaches perform very differently depending on the specific generation of GPU used. On a Tesla C1060 (compute capability 1.3) running the test code (from Github) gives the following results.


```
Device : Tesla C1060

Time for sequential transfer and execute (ms ): 12.92381
  max error : 2.3841858E -07
Time for asynchronous V1 transfer and execute (ms ): 13.63690 
  max error : 2.3841858E -07
Time for asynchronous V2 transfer and execute (ms ): 8.84588
  max error : 2.3841858E -07
```

On a Tesla C2050 (compute capability 2.0) we get the following results.

```
Device : Tesla C2050

Time for sequential transfer and execute (ms ): 9.984512
  max error : 1.1920929e -07
Time for asynchronous V1 transfer and execute (ms ): 5.735584 
  max error : 1.1920929e -07
Time for asynchronous V2 transfer and execute (ms ): 7.597984
  max error : 1.1920929e -07

```

Here the first time reported is the sequential transfer and kernel execution using blocking transfers, which we use as a baseline for asynchronous speedup comparison. 
Why do the two asynchronous strategies perform differently on different architectures? 
To decipher these results we need to understand a bit more about how CUDA devices schedule and execute tasks. 
CUDA devices contain engines for various tasks, which queue up operations as they are issued. 
Dependencies between tasks in different engines are maintained, but within any engine all external dependencies are lost; tasks in each engine's queue are executed in the order they are issued. 
See detail in the original post [here](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/).
