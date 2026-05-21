#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "digits.cuh"

namespace schoolbook_gpu {

__global__ void schoolbook_partials_kernel(
    const int* __restrict__ d_a,
    const int* __restrict__ d_b,
    int* __restrict__ d_partials,
    int n,
    int row_threads) {
    int j = static_cast<int>(blockIdx.x);
    if (j >= n) {
        return;
    }

    __shared__ int row_products[MAX_N];
    __shared__ int row_digits[MAX_N + 1];
    int bj = d_b[j];
    int t = static_cast<int>(threadIdx.x);
    int block_threads = static_cast<int>(blockDim.x);

    for (int i = t; i < n; i += block_threads) {
        row_products[i] = device_digits::MULT_TABLE[d_a[i]][bj];
    }
    __syncthreads();

    if (t == 0) {
        int carry = 0;
        int row_len = 0;
        for (int i = 0; i < n; ++i) {
            int product = row_products[i];
            int total = product + carry;
            int partial_digit = total % 10;
            carry = total / 10;
            row_digits[row_len++] = partial_digit;
        }
        if (carry) {
            row_digits[row_len++] = carry;
        }
        int base = j * (n + 1);
        for (int i = 0; i < row_len; ++i) {
            d_partials[base + i] = row_digits[i];
        }
        for (int i = row_len; i < n + 1; ++i) {
            d_partials[base + i] = 0;
        }
    }
}

__global__ void schoolbook_accumulate_kernel(
    const int* __restrict__ d_partials,
    int* __restrict__ d_result,
    int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int acc[MAX_RESULT_DIGS + MAX_N];
    int acc_len = 1;
    acc[0] = 0;

    for (int j = 0; j < n; ++j) {
        int base = j * (n + 1);
        for (int idx = 0; idx < n + 1; ++idx) {
            int pdigit = d_partials[base + idx];
            if (pdigit) {
                device_digits::add_digit_at(acc, acc_len, j + idx, pdigit);
            }
        }
    }

    for (int i = 0; i < acc_len && i < MAX_RESULT_DIGS; ++i) {
        d_result[i] = acc[i];
    }
    for (int i = acc_len; i < MAX_RESULT_DIGS; ++i) {
        d_result[i] = 0;
    }
}

inline cudaError_t schoolbook_multiply_gpu(
    const int* h_a_digits,
    const int* h_b_digits,
    int n,
    long long* h_result,
    float* elapsed_ms,
    int row_threads) {
    if (n < 1 || n > MAX_N) {
        return cudaErrorInvalidValue;
    }

    int block_threads = row_threads;
    if (block_threads < 1) block_threads = 1;
    if (block_threads > 32) block_threads = 32;

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_partials = nullptr;
    int* d_result = nullptr;

    std::vector<int> h_res(MAX_RESULT_DIGS, 0);
    int out_len = 0;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    float ms = 0.f;

    cudaError_t err = cudaMalloc(&d_a, n * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_b, n * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_partials, n * (n + 1) * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_result, MAX_RESULT_DIGS * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMemcpy(d_a, h_a_digits, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_b, h_b_digits, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    schoolbook_partials_kernel<<<n, block_threads>>>(d_a, d_b, d_partials, n, block_threads);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_events;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup_events;

    schoolbook_accumulate_kernel<<<1, 1>>>(d_partials, d_result, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_events;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup_events;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (elapsed_ms) {
        *elapsed_ms = ms;
    }

    err = cudaMemcpy(h_res.data(), d_result, MAX_RESULT_DIGS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_events;

    out_len = 2 * n;
    while (out_len > 1 && h_res[static_cast<size_t>(out_len - 1)] == 0) {
        out_len--;
    }
    h_res.resize(static_cast<size_t>(out_len));
    *h_result = host_digits::digits_to_int(h_res);

cleanup_events:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
cleanup:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partials);
    cudaFree(d_result);
    return err;
}

inline cudaError_t schoolbook_multiply_gpu(
    long long a,
    long long b,
    long long* h_result,
    float* elapsed_ms,
    int row_threads) {
    if (a == 0 || b == 0) {
        *h_result = 0;
        if (elapsed_ms) *elapsed_ms = 0.f;
        return cudaSuccess;
    }
    int n = host_digits::padded_width(a, b);
    auto da = host_digits::pad_to_length(host_digits::int_to_digits(a), n);
    auto db = host_digits::pad_to_length(host_digits::int_to_digits(b), n);
    return schoolbook_multiply_gpu(
        da.data(), db.data(), n, h_result, elapsed_ms, row_threads);
}

}  // namespace schoolbook_gpu
