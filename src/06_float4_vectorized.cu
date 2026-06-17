#include<iostream>
#include<cuda_runtime.h>
#include<iomanip>
using namespace std;


// 这里面不用cudaSuccess就是这样写 因为成功就是0
#define CUDA_CHECK(call)\
    do{\
        cudaError_t err = call;\
        if (err){\
            cerr << "CUDA Error: " << cudaGetErrorString(err) <<\
            " at " << __FILE__ << ":" << __LINE__ << endl;\
            exit(1);\
        }\
    }while(0)


// 这是单一的版本 也就是一个thread处理一个float
__global__ void elementwise_naive(const float* A, const float* B, float* C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

// 这个就是一下处理4个floats
__global__ void elementwise_float4(const float* A, const float* B, float* C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int n_float = n / 4;

    if (idx < n_float) {
        // 重新解释转换
        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);

        float4 a = A4[idx];
        float4 b = B4[idx];

        float4 c;
        c.x = a.x * b.x;
        c.y = a.y * b.y;
        c.z = a.z * b.z;
        c.w = a.w * b.w;

        C4[idx] = c;
    }

    // 然后处理一下不能整除剩下的数据
    int tail_start = n_float * 4;
    int tail_idx = tail_start + idx;

    if (tail_idx < n) {
        C[tail_idx] = A[tail_idx] * B[tail_idx];
    }
}

float benchmark_naive(const float* d_A, const float* d_B, float* d_C, int n,
     int threadsPerBlock, int repeat) {
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 先自己算一遍
    elementwise_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start)); 

    for (int i=0; i<repeat; ++i) {
        elementwise_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;

}


// C虽然是可变 但同时钥匙也是给了GPU 这一点记住
float benchmark_float4(const float* d_A, const float* d_B, float* d_C,
     int n, int threadsPerBlock, int repeat) {
    int n_float = n / 4;
    int blocksPerGrid = (n_float + threadsPerBlock - 1) / threadsPerBlock;

    // 先自己算一遍
    elementwise_float4<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i=0; i<repeat; ++i) {
        elementwise_float4<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

bool check_result(const float* h_C, int n) {
    for (int i=0; i<n; ++i) {
        if(h_C[i] != 6.0f) {
            cout << "Error at index " << i <<
                ": got " << h_C[i] <<
                ", expected 6.0" << endl;

            return false;
        }
    }
    return true;
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    cout << "Array size: " << n << " elements" << endl;
    cout << "Data type: float32" << endl;
    cout << "Memory per array: " << bytes/1024.0/1024.0 << " MB" << endl;

    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    for (int i=0; i < n; ++i) {
        h_A[i] = 2.0f;
        h_B[i] = 3.0f;
        h_C[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int repeat = 200;

    double bytes_moved = 3*n*sizeof(float);

    // 先进行naive
    float naive_ms = benchmark_naive(d_A, d_B, d_C, n, threadsPerBlock, repeat);

    // 得到结果
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    // 检查结果
    bool naive_correct = check_result(h_C, n);

    // 先清空d_C 避免出现问题
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    // 在进行float4
    float float4_ms = benchmark_float4(d_A, d_B, d_C, n, threadsPerBlock, repeat);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    bool float4_correct = check_result(h_C, n);

    double naive_bw = bytes_moved / (naive_ms / 1000.0) / 1e9;
    double float4_bw = bytes_moved / (float4_ms / 1000.0) / 1e9;

    cout << endl;
    cout << left << setw(15) << "Kernel" <<
        setw(18) << "AvgTime(ms)" << 
        setw(20) << "Bandwidth(GB/s)" <<
        setw(15) << "Check" << endl;

    cout << left << setw(15) << "naive" <<
        setw(18) << naive_ms << 
        setw(20) << naive_bw <<
        setw(15) << (naive_correct ? "PASSED" : "FAILED") << endl;

    cout << left << setw(15) << "float4" <<
        setw(18) << float4_ms << 
        setw(20) << float4_bw <<
        setw(15) << (float4_correct ? "PASSED" : "FAILED") << endl;

    cout << endl;
    cout << "Speedup(float4 over naive): " <<
        naive_ms / float4_ms << "x" << endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));


    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
    

}