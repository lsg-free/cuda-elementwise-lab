Array size: 16777216 elements
Data type: float32
Memory per array: 64 MB

BlockSzie      AvgTime(ms)       Bandwidth(GB/s)     
-----------------------------------------------------
128            0.388805          517.809             
256            0.393396          511.765             
512            0.39226           513.248             
1024           0.401236          501.766             

Result check: PASSED

Array size: 16777216 elements
Data type: float32
Memory per array: 64 MB

BlockSzie      AvgTime(ms)       Bandwidth(GB/s)     
-----------------------------------------------------
128            0.390745          515.238             
256            0.39164           514.06              
512            0.392257          513.252             
1024           0.39923           504.287             

Result check: PASSED

每次跑的结果都是不一样的  但是的话就是大致是一样的 这就可以了
因为这里面也是涉及了很多原因
1. GPU 当前温度
2. GPU 当前功耗状态
3. 显卡频率动态变化
4. WSL 环境调度开销
5. 系统后台进程
6. 第一次/第二次运行时缓存状态不同
7. CUDA kernel launch 和事件计时本身有微小抖动

但是现在主流的还是在进行256 blocksize 也是常见的默认选择




这是一个显存处理一个元素和一个显存处理多个元素的结果
Array size: 16777216 elements
Data type: float32
Memory per array: 64 MB

Kernel         AvgTime(ms)       Bandwidth(GB/s)     Check          
naive          0.394002          510.979             PASSED         
float4         0.390764          515.213             PASSED         

Speedup(float4 over naive): 1.00829x

结果就是差不多   因为说明该 kernel 已经主要受显存带宽限制。

        我们在使用__half时就会用到这几个
        加法：__hadd(a, b)
        减法：__hsub(a, b)
        除法：__hdiv(a, b)
        乘法：__hmul(a, b)
        乘加指令（FMA）：__hfma(a, b, c) -> 算出 a * b + c

Array size: 16777216 elements
Data type: float16/half
Memory per array: 32 MB

Kernel         AvgTime(ms)       Bandwidth(GB/s)     Check          
--------------------------------------------------------------------
half           0.208287          483.29              1              
half2          0.179846          559.72              1              

Sueedup(half2 over half):1.15814x

还是有用的对于half2来说



就是这里面用ai写了一个run.sh 就可以直接用./run.sh 07_half2_vectorized 这样直接跑  不过就是之前要给权限 也就是chmod +x run.sh

之前的话就是用 nvcc 文件 -o 输出名  然后运行就是 ./输出名

但是用AI写了一个run.sh的话就是
./run.sh 01_thread_index 
./run.sh 02_elementwise_mul_naive 
./run.sh 03_elementwise_mul_timing 
./run.sh 04_elementwise_mul_bandwidth 
./run.sh 05_block_size_sweep 
./run.sh 06_float4_vectorized 
./run.sh 07_half2_vectorized

这是09的结果
Data type: FP32
Memory per array: 64 MB
Threads per block: 256
Repeat: 100

Op          AvgTime(ms)       Bandwidth(GB/s)     Result            Expected          Check       
--------------------------------------------------------------------------------------------------
Sum         0.153658          438.452             1.67772e+07       1.67772e+07       PASSED      
Max         0.152695          441.217             499               499               PASSED      

Note: Bandwidth is estimated from global memory traffic across all reduction levels.