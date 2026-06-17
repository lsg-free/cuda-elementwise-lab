#include<iostream>
#include<cuda_runtime.h>
#include<stdio.h>
using namespace std;


__global__ void print_thread_index() {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("blockIdx.x = %d, threadIdx.x = %d, blockDim.x = %d, global id = %d\n",
            bid, tid, block_size, gid);
}


int main() {
    printf("Launch kernel: 2 blocks, 4 threads per block\n");

    print_thread_index<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}