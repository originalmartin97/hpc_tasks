#include <stdio.h>

__global__ void hello_kernel()
{
	int global_x = blockIdx.x * blockDim.x + threadIdx.x;
	int global_y = blockIdx.y * blockDim.y + threadIdx.y;
	int global_z = blockIdx.z * blockDim.z + threadIdx.z;
    printf("Hello from GPU thread (%d, %d, %d) = (%d, %d, %d) * (%d, %d, %d) + (%d, %d, %d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
	global_x, global_y, global_z);
}

int main()
{
    dim3 numThreadsInBlock(4, 3, 2);
    dim3 numBlocks(3, 2, 4);

    hello_kernel<<<numBlocks, numThreadsInBlock>>>();

    cudaDeviceSynchronize();

    return 0;
}
