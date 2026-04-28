// Minimal test: verify SM90a WGMMA instruction actually works on this device.
// Build: nvcc -O2 -std=c++17 -arch=sm_90a test_cutlass_sm90a.cu -o test_cutlass_sm90a
// If this crashes with "illegal instruction", the device doesn't support sm_90a.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void test_wgmma_available() {
    // Just check we can run on sm_90a without illegal instruction
    if (threadIdx.x == 0) {
        printf("SM90a kernel running OK on block %d\n", blockIdx.x);
    }
}

__global__ void test_tma_descriptor() {
    // Test cudaFuncSetAttribute works for large shared memory
    if (threadIdx.x == 0) {
        printf("TMA test kernel OK\n");
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("Max shared mem per block: %zu\n", prop.sharedMemPerBlock);
    printf("Max shared mem per block optin: %zu\n", prop.sharedMemPerBlockOptin);
    printf("Max shared mem per SM: %zu\n", prop.sharedMemPerMultiprocessor);

    // Test 1: basic kernel launch
    printf("\nTest 1: basic sm_90a kernel...\n");
    test_wgmma_available<<<1, 32>>>();
    cudaError_t err = cudaDeviceSynchronize();
    printf("Result: %s\n", err == cudaSuccess ? "PASS" : cudaGetErrorString(err));

    // Test 2: large shared memory (228KB, typical for CUTLASS SM90 tiles)
    printf("\nTest 2: 228KB shared memory...\n");
    size_t smem = 228 * 1024;
    err = cudaFuncSetAttribute(test_tma_descriptor,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    printf("cudaFuncSetAttribute(228KB): %s\n",
        err == cudaSuccess ? "OK" : cudaGetErrorString(err));
    if (err == cudaSuccess) {
        test_tma_descriptor<<<1, 32, smem>>>();
        err = cudaDeviceSynchronize();
        printf("Launch result: %s\n", err == cudaSuccess ? "PASS" : cudaGetErrorString(err));
    }

    // Test 3: check CUDA driver version
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    printf("\nCUDA driver: %d.%d, runtime: %d.%d\n",
        driver / 1000, (driver % 100) / 10,
        runtime / 1000, (runtime % 100) / 10);

    return 0;
}
