#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "digits.cuh"

namespace vedic_gpu {

__global__ void vedic_column_kernel(
    const int* __restrict__ d_a,
    const int* __restrict__ d_b,
    int* __restrict__ d_result,
    const int* __restrict__ d_carry_in,
    int carry_in_len,
    int* __restrict__ d_carry_out,
    int* __restrict__ d_carry_out_len,
    int k,
    int n) {
    __shared__ int products[MAX_N];
    int p = device_digits::column_products(k, n);
    int i_start = k - n + 1;
    if (i_start < 0) {
        i_start = 0;
    }
    int t = static_cast<int>(threadIdx.x);
    int block_threads = static_cast<int>(blockDim.x);

    for (int idx = t; idx < p; idx += block_threads) {
        int i = i_start + idx;
        int j = k - i;
        products[idx] = device_digits::MULT_TABLE[d_a[i]][d_b[j]];
    }
    __syncthreads();

    if (t == 0) {
        int column[MAX_COLUMN_DIGS];
        int col_len = 0;

        for (int idx = 0; idx < p; ++idx) {
            device_digits::add_value_at_position(column, col_len, 0, products[idx]);
        }
        for (int idx = 0; idx < carry_in_len; ++idx) {
            device_digits::add_digit_at(column, col_len, idx, d_carry_in[idx]);
        }

        if (col_len == 0) {
            column[0] = 0;
            col_len = 1;
        }

        d_result[k] = column[0];

        int out_len = col_len > 1 ? col_len - 1 : 0;
        *d_carry_out_len = out_len;
        for (int i = 0; i < out_len; ++i) {
            d_carry_out[i] = column[i + 1];
        }
        for (int i = out_len; i < MAX_CARRY_BUF; ++i) {
            d_carry_out[i] = 0;
        }
    }
}

inline void vedic_flush_carry_host(
    std::vector<int>& result_digits,
    const std::vector<int>& carry_digits,
    int n) {
    for (size_t idx = 0; idx < carry_digits.size(); ++idx) {
        if (carry_digits[idx]) {
            host_digits::add_digit_at(
                result_digits,
                (2 * n - 1) + static_cast<int>(idx),
                carry_digits[idx]);
        }
    }
}

inline cudaError_t vedic_multiply_gpu(
    const int* h_a_digits,
    const int* h_b_digits,
    int n,
    long long* h_result,
    float* elapsed_ms,
    int threads_per_column_block) {
    if (n < 1 || n > MAX_N) {
        return cudaErrorInvalidValue;
    }

    int block_threads = threads_per_column_block;
    if (block_threads < 1) {
        block_threads = 1;
    }
    if (block_threads > 32) {
        block_threads = 32;
    }

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_result = nullptr;
    int* d_carry_a = nullptr;
    int* d_carry_b = nullptr;
    int* d_carry_len = nullptr;

    std::vector<int> h_result_digits;
    std::vector<int> h_carry;
    int h_carry_len = 0;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    float ms = 0.f;

    cudaError_t err = cudaMalloc(&d_a, n * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_b, n * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_result, MAX_RESULT_DIGS * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_carry_a, MAX_CARRY_BUF * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_carry_b, MAX_CARRY_BUF * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_carry_len, sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMemcpy(d_a, h_a_digits, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_b, h_b_digits, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemset(d_result, 0, MAX_RESULT_DIGS * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemset(d_carry_a, 0, MAX_CARRY_BUF * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemset(d_carry_b, 0, MAX_CARRY_BUF * sizeof(int));
    if (err != cudaSuccess) goto cleanup;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int* d_carry_in = d_carry_a;
    int* d_carry_out = d_carry_b;
    h_carry_len = 0;
    err = cudaMemcpy(d_carry_len, &h_carry_len, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_events;

    for (int k = 0; k < 2 * n - 1; ++k) {
        vedic_column_kernel<<<1, block_threads>>>(
            d_a, d_b, d_result, d_carry_in, h_carry_len, d_carry_out, d_carry_len,
            k, n);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup_events;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto cleanup_events;

        err = cudaMemcpy(&h_carry_len, d_carry_len, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto cleanup_events;

        int* tmp = d_carry_in;
        d_carry_in = d_carry_out;
        d_carry_out = tmp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (elapsed_ms) {
        *elapsed_ms = ms;
    }

    h_result_digits.assign(static_cast<size_t>(2 * n - 1), 0);
    err = cudaMemcpy(
        h_result_digits.data(),
        d_result,
        (2 * n - 1) * sizeof(int),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_events;

    h_carry.resize(static_cast<size_t>(h_carry_len));
    if (h_carry_len > 0) {
        err = cudaMemcpy(
            h_carry.data(),
            d_carry_in,
            h_carry_len * sizeof(int),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto cleanup_events;
    }
    vedic_flush_carry_host(h_result_digits, h_carry, n);
    *h_result = host_digits::digits_to_int(h_result_digits);

cleanup_events:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
cleanup:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_carry_a);
    cudaFree(d_carry_b);
    cudaFree(d_carry_len);
    return err;
}

inline cudaError_t vedic_multiply_gpu(
    long long a,
    long long b,
    long long* h_result,
    float* elapsed_ms,
    int threads_per_column_block) {
    if (a == 0 || b == 0) {
        *h_result = 0;
        if (elapsed_ms) *elapsed_ms = 0.f;
        return cudaSuccess;
    }
    int n = host_digits::padded_width(a, b);
    auto da = host_digits::pad_to_length(host_digits::int_to_digits(a), n);
    auto db = host_digits::pad_to_length(host_digits::int_to_digits(b), n);
    return vedic_multiply_gpu(
        da.data(), db.data(), n, h_result, elapsed_ms, threads_per_column_block);
}

}  // namespace vedic_gpu
