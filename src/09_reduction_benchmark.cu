#include<iostream>
#include<iomanip>
#include<cmath>
#include<cuda_runtime.h>
#include<cfloat>
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

__global__ void reduce_sum_kernel(const float* input, float* partial, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    // 这个就是越界的时候加0
    float sum = 0.0f;

    if (idx < n) {
        sum += input[idx];
    }

    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }

    // 这个是重新打好标签
    sdata[tid] = sum;
    __syncthreads();

    // >> 二进制向右移动一位
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// 这个是max的
__global__ void reduce_max_kernel(const float* input, float* partial, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    //无限小
    float max_val = -FLT_MAX;

    if(idx < n) {
        max_val = fmaxf(max_val, input[idx]);
    }

    if (idx + blockDim.x < n) {
        max_val = fmaxf(max_val, input[idx + blockDim.x]);
    }

    sdata[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

//  在进行多轮 reduction：
const float* launch_reduce_sum(const float* d_input, float* d_tmp1, float* d_tmp2, int n, int threads_per_block) {
    const float* d_in = d_input; // 输入
    float* d_out = d_tmp1; // 输出

    int current_n = n;

    while (current_n > 1) {
        int blocks = (current_n + threads_per_block * 2 - 1) / (threads_per_block * 2);
        size_t shared_bytes = threads_per_block * sizeof(float);

        // 这里已经进行了传递 是指针
        reduce_sum_kernel<<<blocks, threads_per_block, shared_bytes>>>(d_in, d_out, current_n);

        current_n = blocks;
        d_in = d_out;
        d_out = (d_out == d_tmp1) ? d_tmp2 : d_tmp1;
    }

    return d_in;

}

const float* launch_reduce_max(const float* d_input, float* d_tmp1, float* d_tmp2, int n, int threads_per_block) {
    const float* d_in = d_input;
    float* d_out = d_tmp1;

    int current_n = n;

    while(current_n > 1) {
        int blocks = (current_n + threads_per_block * 2 - 1) / (threads_per_block * 2);
        size_t shared_bytes = threads_per_block * sizeof(float);

        reduce_max_kernel<<<blocks, threads_per_block, shared_bytes>>>(d_in, d_out, current_n);

        current_n = blocks;
        d_in = d_out;
        d_out = (d_out == d_tmp1) ? d_tmp2 : d_tmp1;
    }

    return d_in;

}

template <typename Launch>
float benchmark_kernel(Launch launcher, int repeat) {
    // warmup
    launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i=0; i < repeat; ++i) {
        launcher();
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

// 这个就是估算每一轮大致是：读 current_n 个 float + 写 blocks 个 partial
double estimate_reduction_bytes(int n, int threads_per_block) {
    int current_n = n;
    double bytes = 0.0;

    while(current_n > 1) {
        int blocks = (current_n + threads_per_block * 2 - 1) / (threads_per_block * 2);

        bytes += current_n * sizeof(float);
        bytes += blocks * sizeof(float);

        current_n = blocks;
    }

    return bytes;
}

void print_result (const string& op, float avg_ms, double bandwidth, float result, float expected, bool passed) {
    cout << left << setw(12) << op
         << setw(18) << avg_ms
         << setw(20) << bandwidth
         << setw(18) << result
         << setw(18) << expected
         << setw(12) << (passed ? "PASSED" : "FAILED")
         << endl;
}

int main() {
    int n = 1 << 24;
    int threads_per_block = 256;
    int repeat = 100;

    size_t bytes = n * sizeof(float);

    cout << "CUDA Reduction Benchmark" << endl;
    cout << "Array size: " << n << " elements" << endl;
    cout << "Data type: FP32" << endl;
    cout << "Memory per array: " << bytes / 1024.0 / 1024.0 << " MB" << endl;
    cout << "Threads per block: " << threads_per_block << endl;
    cout << "Repeat: " << repeat << endl;

    int max_partial_size = (n + threads_per_block * 2 - 1) / (threads_per_block * 2);
    size_t partial_bytes = max_partial_size * sizeof(float);

    // host数据
    float* h_sum_input = new float[n];
    float* h_max_input = new float[n];

    // 先假设全是1吧 sum
    for (int i=0; i < n; ++i) {
        h_sum_input[i] = 1.0f;
    }

    // max 就在特定数值之间
    for (int i=0; i < n; ++i) {
        h_max_input[i] = static_cast<float>((i % 1000) - 500);
    }


    // gpu的内存
    float *d_sum_input, *d_max_input;
    float *d_tmp1, *d_tmp2;

    CUDA_CHECK(cudaMalloc(&d_sum_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_max_input, bytes));

    CUDA_CHECK(cudaMalloc(&d_tmp1, partial_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp2, partial_bytes));

    CUDA_CHECK(cudaMemcpy(d_sum_input, h_sum_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max_input, h_max_input, bytes, cudaMemcpyHostToDevice));

    double reduction_bytes = estimate_reduction_bytes(n, threads_per_block);

    cout << endl;
    cout << left << setw(12) << "Op"
         << setw(18) << "AvgTime(ms)"
         << setw(20) << "Bandwidth(GB/s)"
         << setw(18) << "Result"
         << setw(18) << "Expected"
         << setw(12) << "Check"
         << endl;

    cout << string(98, '-') << endl;

    // sum benchmark
    float sum_ms = benchmark_kernel([&]() {
        launch_reduce_sum(d_sum_input, d_tmp1, d_tmp2, n, threads_per_block);
    }, repeat);

    const float* d_sum_result = launch_reduce_sum(d_sum_input, d_tmp1, d_tmp2, n, threads_per_block);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum_result, d_sum_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 正确性检查
    float expected_sum = static_cast<float>(n);
    bool sum_passed = fabs(h_sum_result - expected_sum) < 1e-2f;

    double sum_bw = reduction_bytes/(sum_ms/1000.0)/1e9;

    print_result("Sum", sum_ms, sum_bw, h_sum_result, expected_sum, sum_passed);

    // max
    float max_ms = benchmark_kernel([&]() {
        launch_reduce_max(d_max_input, d_tmp1, d_tmp2, n, threads_per_block);
    }, repeat);

    const float* d_max_result = launch_reduce_max(d_max_input, d_tmp1, d_tmp2, n, threads_per_block);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_max_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_max_result, d_max_result, sizeof(float), cudaMemcpyDeviceToHost));

    float expected_max = 499.0f;
    bool max_passed = fabs(h_max_result - expected_max) < 1e-5f;

    double max_bw = reduction_bytes / (max_ms / 1000.0) / 1e9;

    print_result("Max", max_ms, max_bw, h_max_result, expected_max, max_passed);

    cout << endl;
    cout << "Note: Bandwidth is estimated from global memory traffic across all reduction levels." << endl;

    // 清理内存
    CUDA_CHECK(cudaFree(d_sum_input));
    CUDA_CHECK(cudaFree(d_max_input));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));

    delete[] h_sum_input;
    delete[] h_max_input;

    return 0;
}