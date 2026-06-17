#include<iostream>
#include<iomanip>
#include<cmath>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
using namespace std;

#define CUDA_CHECK(call)\
    do {\
    cudaError_t err = call;\
    if (err) {\
        cerr << "CUDA Error: " << cudaGetErrorString(err)\
        << " at " << __FILE__ << ":" << __LINE__ << endl;\
        exit(1);\
    }\
    }while(0)

// half naive: 一个线程处理一个half
__global__ void elementwise_half_naive(const __half* A, const __half* B, __half* C,
     int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        C[idx] = __hmul(A[idx], B[idx]);
    }
}

// half2: 一个线程处理两个 half
__global__ void elementwise_half2(const __half* A, const __half* B, __half* C,
     int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int n_half2 = n/2;

    if(idx < n_half2) {
        // 重新解释转换
        const __half2* A2 = reinterpret_cast<const __half2*>(A);
        const __half2* B2 = reinterpret_cast<const __half2*>(B);
        __half2* C2 = reinterpret_cast<__half2*>(C);

        __half2 a = A2[idx];
        __half2 b = B2[idx];

        C2[idx] = __hmul2(a, b);
    }

    // 如果n是奇数，就单独处理一个half
    int tail_start = n_half2 * 2;
    int tail_idx = tail_start + idx;

    if(tail_idx < n) {
        C[tail_idx] = __hmul(A[tail_idx], B[tail_idx]);
    }
}

float benchmark_half_naive(const __half* d_A, const __half* d_B, __half* d_C,
                            int n, int threadsPerBlock, int repeat) {
    int blocksPerGrid = (n + threadsPerBlock + 1) / threadsPerBlock;
    
    // 先运行一次
    elementwise_half_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i=0; i<repeat; ++i) {
        elementwise_half_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms/repeat;
    }

float benchmark_half2(const __half* d_A,const __half* d_B, __half* d_C,
                        int n, int threadsPerBlock, int repeat) {
    int n_half2 = n/2;
    int blocksPerGrid = (n_half2 + threadsPerBlock - 1) / threadsPerBlock;

    // 先运行一次
    elementwise_half2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i=0; i<repeat; ++i) {
        elementwise_half2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return total_ms/repeat;
}

bool check_result(const __half* h_C, int n) {
    for(int i=0; i<n; ++i) {
        float value = __half2float(h_C[i]);  // 这个是转换出来也就是半精度转32位    h_A[i] = __float2half(2.0f); 这个就是压缩
        if (fabs(value - 6.0f) > 1e-3f) {   // 这里面的fabs是浮点绝对值  Float Absolute Value
            cout << "Error at index " << i
                << ": got " << value <<
                ", expected 6.0" << endl;
            return false;
        }
    }
    return true;
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(__half);

    cout << "Array size: " << n << " elements" << endl;
    cout << "Data type: float16/half" << endl;
    cout << "Memory per array: " << bytes / 1024.0 / 1024.0 << " MB" << endl;

    __half* h_A = new __half[n];
    __half* h_B = new __half[n];
    __half* h_C = new __half[n];

    for (int i=0; i<n; ++i) {
        h_A[i] = __float2half(2.0f);
        h_B[i] = __float2half(3.0f);
        h_C[i] = __float2half(0.0f);
    }

    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int repeat = 200;

    // 这里面换了就是现在是2个字节
    double bytes_moved = 3*n*sizeof(__half);

    // half naive
    float half_naive_ms = benchmark_half_naive(d_A, d_B, d_C, n, threadsPerBlock, repeat);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    bool half_naive_correct = check_result(h_C, n);

    // 清空
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    float half2_ms = benchmark_half2(d_A, d_B, d_C, n, threadsPerBlock, repeat);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    bool half2_corrent = check_result(h_C, n);

    double half_naive_bw = bytes_moved / (half_naive_ms / 1000.0) / 1e9;
    double half2_bw = bytes_moved / (half2_ms / 1000.0) / 1e9;

    cout << endl;
    cout << left << setw(15) << "Kernel" <<
        setw(18) << "AvgTime(ms)" <<
        setw(20) << "Bandwidth(GB/s)" <<
        setw(15) << "Check" << endl;

    cout << string(68, '-') << endl;
    cout << left << setw(15) << "half" <<
        setw(18) << half_naive_ms <<
        setw(20) << half_naive_bw <<
        setw(15) << half_naive_correct << endl;

    cout << left << setw(15) << "half2" <<
        setw(18) << half2_ms <<
        setw(20) << half2_bw <<
        setw(15) << half2_corrent << endl;

    cout << endl;
    cout << "Speedup(half2 over half):" <<
        half_naive_ms / half2_ms << "x" << endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}