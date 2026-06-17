#include<iostream>
#include<cuda_runtime.h>
using namespace std;

// 错误检查宏
#define CUDA_CHECK(call)\
    do{\
        cudaError_t err = call;\
        if (err != cudaSuccess) {\
            cerr << "CUDA Error: " << cudaGetErrorString(err)\
            << " at " << __FILE__ << ":" << __LINE__ << endl;\
            exit(1);\
        }\
    } while(0)

__global__ void elementwise(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        C[idx] = A[idx] * B[idx];
    }
}

int main() {
    // 先设置一下有多少个元素
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    cout << "Array size: " << n << " elements" << endl;

    // 内存分配在cpu里面  h_指cpu也就是主机 d_指gpu也就是设备
    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    // 开始初始化cpu数据
    for (int i=0; i<n; ++i) {
        h_A[i] = 2.0f;
        h_B[i] = 3.0f;
    }

    // 开始分配gpu内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 开始传递 将cpu传递到gpu里面
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 开始配置kernel   这个就可以计算出有多少blocks 这里面就是2**20/256
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Launch config: " << blocksPerGrid
        << " blocks, " << threadsPerBlock
        << " threads per block" << endl;

    // 启动核心 kernel
    elementwise<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 输出 gpu->cpu
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 检查输出的结果
    bool correct = true;
    for (int i=0; i < n; ++i) {
        if (h_C[i] != 6.0f) {
            correct = false;
            cout << "Error at index " << i
            << ": got" << h_C[i] <<
            ", expected 6.0" << endl;
        break;
        }
    }

    if (correct) {
        cout << "Result check: PASSED" << endl;
    } else {
        cout << "Result check: FAILED" << endl;
    }

    // 首先输出前面十个结果看一下
    cout << "10 results: " << endl;
    for (int i=0; i < 10; ++i) {
        cout << "C[" << i << "] = " << h_C[i] << endl;
    }

    // 最后就是释放内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;

}