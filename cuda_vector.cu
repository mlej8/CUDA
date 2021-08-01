#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

__global__ void pass_vector(int *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] += 5;
    }
}

int main(void)
{
    // H has storage for 4 integers
    thrust::host_vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;

    // H.size() returns the size of vector H
    std::cout << "H has size " << H.size() << std::endl;

    // print contents of H
    for (int i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // resize H
    H.resize(2);

    std::cout << "H now has size " << H.size() << " and its content is:" << std::endl;
    // print contents of H
    for (int i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // Copy host_vector H to device_vector D
    thrust::device_vector<int> D = H;

    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;

    int n = 5;
    int *t;
    cudaMallocManaged(&t, n * sizeof(int));

    std::cout << "before kernel: " << std::endl;
    for (int i = 0; i < n; i++)
    {
        t[i] = i;
        std::cout << t[i] << " " << std::endl;
    }
    pass_vector<<<1, 1>>>(t, n);
    cudaDeviceSynchronize();
    std::cout << "after kernel: " << std::endl;
    for (int i = 0; i < n; i++)
    {
        std::cout << t[i] << " " << std::endl;
    }

    // print contents of D
    for (int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // H and D are automatically deleted when the function returns
    return 0;
}