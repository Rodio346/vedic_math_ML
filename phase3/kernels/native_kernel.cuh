#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "digits.cuh"

namespace native_gpu {

__global__ void native_multiply_kernel(long long a, long long b, long long* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = a * b;
    }
}

inline cudaError_t native_multiply_gpu(
    long long a,
    long long b,
    long long* h_result,
    float* elapsed_ms) {
    long long* d_result = nullptr;
    cudaError_t err = cudaMalloc(&d_result, sizeof(long long));
    if (err != cudaSuccess) return err;

    cudaEvent_t start{};
    cudaEvent_t stop{};
    float ms = 0.f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    native_multiply_kernel<<<1, 1>>>(a, b, d_result);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        if (elapsed_ms) {
            *elapsed_ms = ms;
        }
        err = cudaMemcpy(h_result, d_result, sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_result);
    return err;
}

// Includes host digit conversion overhead for fair comparison with digit kernels.
inline cudaError_t native_multiply_gpu_with_digits(
    long long a,
    long long b,
    long long* h_result,
    float* elapsed_ms) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    auto da = host_digits::int_to_digits(a);
    auto db = host_digits::int_to_digits(b);
    int n = host_digits::padded_width(a, b);
    da = host_digits::pad_to_length(da, n);
    db = host_digits::pad_to_length(db, n);

    long long product = 0;
    cudaError_t err = native_multiply_gpu(a, b, &product, nullptr);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return err;
    }

    std::vector<int> out_digits = host_digits::int_to_digits(product);
    long long reconstructed = host_digits::digits_to_int(out_digits);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    if (elapsed_ms) {
        *elapsed_ms = ms;
    }
    *h_result = reconstructed;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return cudaSuccess;
}

}  // namespace native_gpu
