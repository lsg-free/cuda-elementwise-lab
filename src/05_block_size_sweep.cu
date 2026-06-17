#include<iostream>
#include<iomanip>  // 这个是输入/输出操纵器  1.控制小数位数 2. std::setw(n) 3. std::setfill('c')：改变填充字符
#include<cuda_runtime.h>
using namespace std;

#define CUDA_CHECK(call)\
    do{\
        cudaError_t err = call;\
        if (err != cudaSuccess){\
            cerr << "CUDA Error: " << cudaGetErrorString(err)\
                << " at " << __FILE__ << ":" << __LINE__ << endl;\
            exit(1);\
        }\
    }while(0)

__global__ void elementwise_naive (const float* A, const float* B, float* C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx]*B[idx];
    }
}

float benchmark_kernel(const float* d_A, const float* d_B, float* d_C,
                        int n, int threadsPerBlock, int repeat) {
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // warmup
    elementwise_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i<repeat; ++i) {
        elementwise_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
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

int main() {
    int n = 1 << 24; // 给多点 稳定点  还有就是这两是常量 所以是计算不是平移
    size_t bytes = n*sizeof(float);  // 主要是给cuda申请内存用的

    cout << "Array size: " << n << " elements" << endl;
    cout << "Data type: float32" << endl;
    cout << "Memory per array: " << bytes/1024.0/1024.0 << " MB" << endl;

    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    for (int i=0; i<n; ++i) {
        h_A[i] = 2.0f;
        h_B[i] = 3.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int repeat = 200;
    int block_sizes[] = {128, 256, 512, 1024};

    double bytes_moved = 3*n*sizeof(float);

    //  这里面是为了输出打印工整
    cout << endl;
    cout << left << setw(15) << "BlockSzie" <<
        setw(18) << "AvgTime(ms)" << 
        setw(20) << "Bandwidth(GB/s)" << endl;

    cout << string(53, '-') << endl;

    for (int block_size : block_sizes) {
        float avg_ms = benchmark_kernel(d_A, d_B, d_C, n, block_size, repeat);

        double avg_seconds = avg_ms/1000.0;
        double Bandwidth_GB_s = bytes_moved/avg_seconds/1e9;

        cout << left << setw(15) << block_size <<
            setw(18) << avg_ms <<
            setw(20) << Bandwidth_GB_s <<
            endl;
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i=0; i<n; ++i) {
        if (h_C[i] != 6.0f) {
            correct = false;
            cout << "Error at index " << i <<
            ": got " << h_C[i] <<
            ", expected 6.0" << endl;

            break;
        }
    }

    cout << endl;
    cout << "Result check: " << (correct ? "PASSED" : "FAILED") << endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_C;
    delete[] h_B;

    return 0;

}