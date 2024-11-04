# Optimize vector addition with CUDA

We start with [add.cpp](https://github.com/RenkeHuang/add_arrays_gpu/blob/main/add.cpp) that adds the elements of two arrays with a million elements each.

Compile and run this C++ program with:
```bash
g++ add.cpp -o add
./add
```
It prints that there was no error in the summation and then exits. 

To turn a normal C++ function to a CUDA kernel, add the specifier `__global__` to the function. This tells the CUDA C++ compiler that this function can be run on the GPU and can be called from CPU. 

### Allocate memory
[Unified Memory](https://developer.nvidia.com/blog/unified-memory-in-cuda-6/) in CUDA provides a single memory space accessible by all GPUs and CPUs in your system. To allocate data in unified memory, we use `cudaMallocManaged()`, which returns a pointer that can be accessed from the host (CPU) code or device (GPU) code. To free the data, pass the pointer to `cudaFree()`. 

### Threads, Blocks, Grid
#### Single-thread
The single-threaded CUDA program for adding two arrays: [add.cu](https://github.com/RenkeHuang/add_arrays_gpu/blob/main/add.cu)
```bash
nvcc add.cu -o add_1thread
./add_1thread
```
Profile this first CUDA kernel by `nvprof`, the GPU profiler in the CUDA Toolkit:
```
nvprof ./add_1thread
```
The output shows a single call to the CUDA kernel [`add`](https://github.com/RenkeHuang/add_arrays_gpu/blob/554067d572772e22117f800f9c1e2c419f2262eb/add.cu#L8-L12), example timings on a `Tesla T4` GPU:
```
==828== NVPROF is profiling process 828, command: ./add_1thread
Max error: 0
==828== Profiling application: ./add_1thread
==828== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  108.30ms         1  108.30ms  108.30ms  108.30ms  add(int, float*, float*)
      API calls:   65.11%  204.16ms         2  102.08ms  39.535us  204.12ms  cudaMallocManaged
                   34.54%  108.31ms         1  108.31ms  108.31ms  108.31ms  cudaDeviceSynchronize
                    0.18%  554.59us         2  277.30us  227.74us  326.85us  cudaFree
                    0.11%  336.37us         1  336.37us  336.37us  336.37us  cudaLaunchKernel
                    0.05%  142.71us       114  1.2510us     146ns  52.506us  cuDeviceGetAttribute
                    0.01%  16.381us         1  16.381us  16.381us  16.381us  cuDeviceGetPCIBusId
                    0.00%  11.894us         1  11.894us  11.894us  11.894us  cuDeviceGetName
                    0.00%  4.6610us         1  4.6610us  4.6610us  4.6610us  cuDeviceTotalMem
                    0.00%  1.2520us         3     417ns     182ns     735ns  cuDeviceGetCount
                    0.00%  1.1060us         2     553ns     240ns     866ns  cuDeviceGet
                    0.00%     489ns         1     489ns     489ns     489ns  cuModuleGetLoadingMode
                    0.00%     340ns         1     340ns     340ns     340ns  cuDeviceGetUuid

==828== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  806.4530us  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  362.5240us  Device To Host
      12         -         -         -           -  2.808599ms  Gpu page fault groups
Total CPU Page faults: 36
```
Check the available GPUs by 
```
nvidia-smi
```
#### Multi-thread in one block
The performance can be improved by utilizing multiple threads within a thread block, check [add_block.cu](https://github.com/RenkeHuang/add_arrays_gpu/blob/main/add_block.cu). The major changes involve [adjusting the **execution configuration**](https://github.com/RenkeHuang/add_arrays_gpu/blob/554067d572772e22117f800f9c1e2c419f2262eb/add_block.cu#L32-L36), and [modifying the indexing of elements in the kernel function](https://github.com/RenkeHuang/add_arrays_gpu/blob/554067d572772e22117f800f9c1e2c419f2262eb/add_block.cu#L11-L13).

The profile result for the updated `add` kernel:
```
==1010== NVPROF is profiling process 1010, command: ./add_block
Max error: 0
==1010== Profiling application: ./add_block
==1010== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  3.7044ms         1  3.7044ms  3.7044ms  3.7044ms  add(int, float*, float*)
      API calls:   79.67%  213.80ms         2  106.90ms  42.090us  213.75ms  cudaMallocManaged
                   18.69%  50.156ms         1  50.156ms  50.156ms  50.156ms  cudaLaunchKernel
                    1.38%  3.7112ms         1  3.7112ms  3.7112ms  3.7112ms  cudaDeviceSynchronize
                    0.19%  523.14us         2  261.57us  231.99us  291.15us  cudaFree
                    0.05%  146.44us       114  1.2840us     160ns  57.327us  cuDeviceGetAttribute
                    0.00%  12.584us         1  12.584us  12.584us  12.584us  cuDeviceGetName
                    0.00%  6.7790us         1  6.7790us  6.7790us  6.7790us  cuDeviceGetPCIBusId
                    0.00%  5.2830us         1  5.2830us  5.2830us  5.2830us  cuDeviceTotalMem
                    0.00%  1.5630us         3     521ns     219ns  1.0000us  cuDeviceGetCount
                    0.00%     959ns         2     479ns     199ns     760ns  cuDeviceGet
                    0.00%     827ns         1     827ns     827ns     827ns  cuModuleGetLoadingMode
                    0.00%     266ns         1     266ns     266ns     266ns  cuDeviceGetUuid

==1010== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  808.5940us  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  362.1710us  Device To Host
      12         -         -         -           -  2.096543ms  Gpu page fault groups
Total CPU Page faults: 36
```
This is a significant speedup from `108.30 ms` to `3.7044 ms` when increasing from one thread to 256 threads.

#### Thread Blocks
CUDA GPUs have parallel processors grouped into Streaming Multiprocessors (SMs) and each SM can run multiple concurrent thread blocks. 
We can optimize the `add` kernel further by launching it with with multiple thread blocks, see [add_grid.cu](https://github.com/RenkeHuang/add_arrays_gpu/blob/main/add_grid.cu).

The syntax of the execution configuration
```cpp
add<<<numBlocks, blockSize>>>(N, x, y);
```

The figure shows the approach to indexing into an one-dimensional array in CUDA via `blockDim.x`, `gridDim.x`, and `threadIdx.x`: the global index of each thread is obtained by computing the offset to the beginning of its block (`blockIdx.x * blockDim.x`) and adding the thread’s index within the block (`threadIdx.x`).
For arrays with one million elements, we use 4096 thread blocks as shown in:
![Find the thread index in a grid](https://developer-blogs.nvidia.com/wp-content/uploads/2017/01/cuda_indexing.png)

Profiling the multi-block kernel gets:
```
==5964== NVPROF is profiling process 5964, command: ./add_grid
Max error: 0
==5964== Profiling application: ./add_grid
==5964== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  2.9296ms         1  2.9296ms  2.9296ms  2.9296ms  add(int, float*, float*)
      API calls:   83.53%  231.19ms         2  115.60ms  49.546us  231.14ms  cudaMallocManaged
                   15.15%  41.919ms         1  41.919ms  41.919ms  41.919ms  cudaLaunchKernel
                    1.04%  2.8810ms         1  2.8810ms  2.8810ms  2.8810ms  cudaDeviceSynchronize
                    0.22%  598.95us         2  299.48us  250.37us  348.58us  cudaFree
                    0.05%  145.43us       114  1.2750us     163ns  57.464us  cuDeviceGetAttribute
                    0.00%  12.744us         1  12.744us  12.744us  12.744us  cuDeviceGetName
                    0.00%  5.4310us         1  5.4310us  5.4310us  5.4310us  cuDeviceGetPCIBusId
                    0.00%  5.3930us         1  5.3930us  5.3930us  5.3930us  cuDeviceTotalMem
                    0.00%  1.8830us         3     627ns     238ns  1.2970us  cuDeviceGetCount
                    0.00%  1.2710us         2     635ns     209ns  1.0620us  cuDeviceGet
                    0.00%     482ns         1     482ns     482ns     482ns  cuModuleGetLoadingMode
                    0.00%     370ns         1     370ns     370ns     370ns  cuDeviceGetUuid

==5964== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      78  105.03KB  4.0000KB  988.00KB  8.000000MB  900.1450us  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  360.5050us  Device To Host
      11         -         -         -           -  2.866382ms  Gpu page fault groups
Total CPU Page faults: 36
```

For some reason, using 4096 blocks does not improve performance to the same level as increasing the number of threads within a single block.
Possible reasons for limited speedup from multiple blocks:
- **Memory Bandwidth Saturation**: With element-wise operations on a large array, your performance is often limited by memory bandwidth rather than compute capability. The Tesla T4 has a finite memory bandwidth, and with 256 threads per block, you may already be reaching close to that limit. Adding more blocks won't reduce the time significantly since memory transfer speeds are the bottleneck.
- **Occupancy and Latency Hiding**: CUDA kernels benefit from having enough threads to keep the GPU fully occupied. However, there’s a threshold where adding more blocks doesn't increase occupancy meaningfully, as the GPU’s resources (e.g., streaming multiprocessors, registers, etc.) are already well-utilized.
- **Thread Divergence and Warp Utilization**: CUDA works best with fully occupied warps (32 threads executing the same instruction simultaneously). With 256 threads per block, you're using exactly 8 warps, which aligns well with GPU architecture. Increasing the number of blocks doesn’t necessarily mean the work will execute faster since the GPU can handle a limited number of concurrent threads.
- **Kernel Launch Overhead**: Launching more blocks introduces a bit of overhead due to scheduling. For simple operations like element-wise addition, this overhead might offset some of the potential gains from additional blocks.
