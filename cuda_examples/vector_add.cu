// =============================================================================
// CUDA EXAMPLE 6 - YOUR FIRST CUDA KERNEL (written in CUDA C++)
// =============================================================================
// The Python examples let PyTorch talk to CUDA for us. This is what is
// underneath: a KERNEL is a small function that thousands of GPU threads run
// at the same time, each one on a different piece of the data.
//
// Compile and run:
//     nvcc vector_add.cu -o vector_add
//     ./vector_add
//
// (If "nvcc: command not found":  export PATH=/usr/local/cuda/bin:$PATH)
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// Stop with a clear message if any CUDA call fails
#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// THE KERNEL: runs on the GPU. "__global__" means "the CPU can launch this".
// Every thread runs this same code; each works out WHICH number is its job.
// ---------------------------------------------------------------------------
__global__ void add(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // my position in the whole grid
    if (i < n) {                                     // the last block may have spare threads
        c[i] = a[i] + b[i];                          // my ONE addition
    }
}

int main()
{
    const int n = 1 << 24;                           // 16,777,216 numbers
    const size_t bytes = n * sizeof(float);

    // Unified memory: on Jetson the CPU and GPU share the same RAM,
    // so one pointer works on both sides.
    float *a, *b, *c;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    // ---- 1. CPU: one worker does all 16 million additions, one after another ----
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- 2. GPU: split the work into blocks of 256 threads ----
    int threads = 256;                                // threads per block
    int blocks = (n + threads - 1) / threads;         // enough blocks to cover every number
    printf("Launching %d blocks x %d threads = %d GPU threads\n", blocks, threads, blocks * threads);

    add<<<blocks, threads>>>(a, b, c, n);             // warm-up launch
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < n; i++) c[i] = 0.0f;          // clear the answers before the timed run

    cudaEvent_t start, stop;                          // GPU stopwatch
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    add<<<blocks, threads>>>(a, b, c, n);             // <<< blocks, threads >>> = launch!
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));                // wait for the GPU to finish
    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // ---- 3. Check every answer (1 + 2 should be 3) ----
    int errors = 0;
    for (int i = 0; i < n; i++) if (c[i] != 3.0f) errors++;

    printf("\nAdded %d pairs of numbers\n", n);
    printf("  CPU : %8.2f ms\n", cpu_ms);
    printf("  GPU : %8.2f ms   (%.1fx faster)\n", gpu_ms, cpu_ms / gpu_ms);
    printf("  Check: %s\n", errors == 0 ? "all answers correct!" : "WRONG ANSWERS");

    cudaFree(a); cudaFree(b); cudaFree(c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return errors == 0 ? 0 : 1;
}
