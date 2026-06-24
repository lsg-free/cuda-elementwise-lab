#include<iostream>
#include<iomanip>
#include<cmath>
#include<cfloat>
#include<cuda_runtime.h>
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

//block做max reduction和sum reduction  rows指行， cols指列
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    //找到当前 block 负责的那一行，在一维数组里的起始地址
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // 先进行求解local max
    float local_max = -FLT_MAX;

    for(int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_input[col]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    // 然后在block内 max reducation
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = sdata[0];

    //计算 exp(x - max)，同时求 local sum
    float local_sum = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        float val = expf(row_input[col] - row_max);
        row_output[col] = val;
        local_sum += val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    //block 内 sum reduction
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = sdata[0];

    // 计算
    for(int col = tid; col < cols; col += blockDim.x) {
        row_output[col] = row_output[col] / row_sum;
    }

}

template<typename Launcher>
float benchmark_kernel(Launcher launcher, int repeat) {
    // 先运行一次
    launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for(int i=0; i<repeat; ++i) {
        launcher();
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

// 检查softmax总和是不是接近1
bool check_softmax_result(const float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;

        for (int c = 0; c < cols; ++c) {
            float val = output[r * cols + c];

            if (!isfinite(val)) {
                cout << "Error: non-finite value at row " << r
                     << ", col " << c << ": " << val << endl;
                return false;
            }

            if (val < -1e-6f) {
                cout << "Error: negative softmax value at row " << r
                     << ", col " << c << ": " << val << endl;
                return false;
            }

            sum += val;
        }

        if (fabs(sum - 1.0) > 1e-3) {
            cout << "Error: row " << r
                 << " sum = " << sum
                 << ", expected 1.0" << endl;
            return false;
        }
    }

    return true;
}

int main() {
    int rows = 4096;
    int cols = 1024;
    int n = rows * cols;

    int threads_per_block = 256;
    int repeat = 100;

    size_t bytes = n * sizeof(float);

    cout << "CUDA Row-wise Softmax Benchmark" << endl;
    cout << "Shape: " << rows << " x " << cols << endl;
    cout << "Total elements: " << n << endl;
    cout << "Data type: FP32" << endl;
    cout << "Input memory: " << bytes / 1024.0 / 1024.0 << " MB" << endl;
    cout << "Output memory: " << bytes / 1024.0 / 1024.0 << " MB" << endl;
    cout << "Threads per block: " << threads_per_block << endl;
    cout << "Repeat: " << repeat << endl;

    float* h_input = new float[n];
    float* h_output = new float[n];

    // 初始话输入 就是相当于构造
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            h_input[idx] = static_cast<float>((c % 128) - 64) / 32.0f;
            h_output[idx] = 0.0f;
        }
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    dim3 grid(rows);  // 启动多少个 block  也就是一个block处理一行
    dim3 block(threads_per_block);  // 相当于一个block里面有多少线程
    size_t shared_bytes = threads_per_block * sizeof(float);

    // 计算
    float avg_ms = benchmark_kernel([&]() {
        softmax_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, rows, cols);
    }, repeat);

    // 导出
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool correct = check_softmax_result(h_output, rows, cols);

    double bytes_moved = 5.0 * n * sizeof(float);
    double bandwidth = bytes_moved / (avg_ms / 1000.0) / 1e9;

    cout << endl;
    cout << left << setw(15) << "Kernel"
         << setw(18) << "AvgTime(ms)"
         << setw(20) << "Bandwidth(GB/s)"
         << setw(12) << "Check"
         << endl;

    cout << string(65, '-') << endl;

    cout << left << setw(15) << "Softmax"
         << setw(18) << avg_ms
         << setw(20) << bandwidth
         << setw(12) << (correct ? "PASSED" : "FAILED")
         << endl;

    cout << endl;
    cout << "Sample output of row 0, first 10 values:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << h_output[i] << " ";
    }
    cout << endl;

    cout << endl;
    cout << "Note: Bandwidth is an approximate estimate based on global memory traffic." << endl;
    cout << "Softmax uses max reduction, exp, sum reduction, and normalization per row." << endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_input;
    delete[] h_output;

    return 0;
}