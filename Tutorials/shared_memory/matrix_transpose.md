# [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
The task in this tutorial is to optimize a matrix transpose to show how to use shared memory to reorder strided global memory accesses into coalesced accesses.

## Matrix Transpose
The code we wish to optimize is a transpose of a matrix of single precision values that operates out-of-place, i.e. the input and output are separate arrays in memory. For simplicity of presentation, we'll consider only square matrices whose dimensions are integral multiples of 32 on a side. The entire code is [available on Github](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu). It consists of several kernels as well as host code to perform typical tasks such as allocation and data transfers between host and device, launches and timing of several kernels as well as validation of their results, and deallocation of host and device memory. In this post I'll only include the kernel code; you can view the rest or try it out on Github.

In addition to performing several different matrix transposes, we run simple matrix copy kernels because copy performance indicates the performance that we would like the matrix transpose to achieve. For both matrix copy and transpose, the relevant performance metric is effective bandwidth, calculated in *GB/s* by dividing twice the size in *GB* of the matrix (once for loading the matrix and once for storing) by time in seconds of execution.

All kernels in this study launch blocks of 32×8 threads (`TILE_DIM=32, BLOCK_ROWS=8 in the code`), and each thread block transposes (or copies) a tile of size 32×32. **Using a thread block with fewer threads than elements in a tile is advantageous for the matrix transpose because each thread transposes four matrix elements**, so much of the index calculation cost is amortized over these elements.

The kernels in this example map threads to matrix elements using a Cartesian `(x,y)` mapping rather than a row/column mapping to simplify the meaning of the components of the automatic variables in CUDA C: `threadIdx.x` is horizontal and `threadIdx.y` is vertical. This mapping is up to the programmer; the important thing to remember is that to ensure memory coalescing we want to map the quickest varying component to contiguous elements in memory.

## Simple Matrix Copy
Let's start by looking at the matrix copy kernel.

```
__global__ void copy(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
}
```
Each thread copies four elements of the matrix in a loop at the end of this routine because the number of threads in a block is smaller by a factor of four (`TILE_DIM/BLOCK_ROWS`) than the number of elements in a tile. Note also that `TILE_DIM` must be used in the calculation of the matrix index `y` rather than `BLOCK_ROWS` or `blockDim.y`. The loop iterates over the second dimension and not the first so that contiguous threads load and store contiguous data, and all reads from idata and writes to odata are coalesced.

**TODO: finish notes**


**By making the tile 33 elements wide rather than 32, we avoid banking conflicts, because elements on the same column no longer belong to the same bank (the 33rd element is now the 2nd word of bank 0, therefore first element of 2nd row is now 2nd word of bank 1, so on). This means that all the elements on the same column do not belong to the same bank.**



