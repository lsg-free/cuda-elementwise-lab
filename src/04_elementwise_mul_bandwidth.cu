#include<iostream>
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

__global__ void elementwise_naive(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    cout << "Array size: " << n << " elements" << endl;

    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    for (int i = 0; i<n; ++i) {
        h_A[i] = 2.0f;
        h_B[i] = 3.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock -1) / threadsPerBlock;

    cout << "Launch config: " << blocksPerGrid
        << " blocks, " << threadsPerBlock <<
        " thread per block" << endl;


    // 先运行一次 防止有其他额外的时间开销
    elementwise_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());  //检查“启动阶段”
    CUDA_CHECK(cudaDeviceSynchronize());  //检查“执行阶段”

    // 创建 CUDA event
    cudaEvent_t start, stop;  // 这是cuda的一个数据类型
    CUDA_CHECK(cudaEventCreate(&start)); // 这是这个数据类型初始化
    CUDA_CHECK(cudaEventCreate(&stop));


    // 这里面多跑几组 让结果更准确了
    int repeat = 100;
    CUDA_CHECK(cudaEventRecord(start));

    for (int i=0; i < repeat; ++i) {
        elementwise_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    }

    // 然后结束时间
    CUDA_CHECK(cudaEventRecord(stop));
    // CPU 阻塞函数 就是看GPU超过了STOP
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    // 现在知道的一般都是用来计时的
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    // 平均时长
    float avg_ms = total_ms / repeat;

    // bandwidth计算
    // C[i] = A[i] * B[i]
    // 每个元素：读 A + 读 B + 写 C = 3 * sizeof(float)
    // 然后在计算带宽
    double bytes_moved = 3.0*n*sizeof(float);
    double avg_seconds = avg_ms/1000.0;
    double bandwidth_GB_s = bytes_moved/avg_seconds/1e9;





    cout << "Total kernel time for " << repeat << " runs "
        << total_ms << " ms" << endl;
    cout << "Average kernel time: " << avg_ms <<
        " ms" << endl;
    cout << "Bytes moved per kernel: "
         << bytes_moved << " bytes" << endl;
    cout << "Effective bandwidth: "
         << bandwidth_GB_s << " GB/s" << endl;


    // 传回去 就是cpu
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 检查输出结果  这个好检查点就写一下
    bool correct = true;
    for (int  i=0; i < n; ++i) {
        if (h_C[i] != 6.0f) {
            correct = false;
            cout << "Error at index " << i <<
                ": got " << h_C[i] <<
                ", expected 6.0" << endl;
            break;
        }
    }

    if (correct) {
        cout << "Result check: PASSED" << endl;
    } else {
        cout << "Result check: FAILED" << endl;
    }

    cout << "10 results: " << endl;
    for (int i=0; i < 10; ++i) {
        cout << "C[" << i << "] = " << h_C[i] << endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;


}