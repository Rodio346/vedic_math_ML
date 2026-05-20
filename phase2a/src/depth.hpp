#pragma once
// Theoretical DAG depth metrics (ported from depth.py).

#include <cmath>

struct DepthMetrics {
    int n_digits;
    int parallel_width;
    int sequential_depth;
    double parallelism_score;
};

inline int column_products(int k, int n) {
    int left = k + 1;
    int right = 2 * n - 1 - k;
    return left < right ? left : right;
}

inline int clamp_n(int n_digits) {
    return n_digits < 1 ? 1 : n_digits;
}

inline DepthMetrics vedic_depth_metrics(int n_digits) {
    int n = clamp_n(n_digits);
    int parallel_width = n;
    int sequential_depth = 0;

    for (int k = 0; k < 2 * n - 1; ++k) {
        int p = column_products(k, n);
        sequential_depth += 1;
        if (p > 1) {
            sequential_depth += static_cast<int>(std::ceil(std::log2(static_cast<double>(p))));
        }
        if (p * 81 >= 10) {
            sequential_depth += 1;
        }
    }

    double score = sequential_depth ? static_cast<double>(parallel_width) / sequential_depth : 0.0;
    return {n, parallel_width, sequential_depth, std::round(score * 10000.0) / 10000.0};
}

inline DepthMetrics schoolbook_depth_metrics(int n_digits) {
    int n = clamp_n(n_digits);
    int parallel_width = n * n;
    int sequential_depth = n + (n - 1);
    double score = sequential_depth ? static_cast<double>(parallel_width) / sequential_depth : 0.0;
    return {n, parallel_width, sequential_depth, std::round(score * 10000.0) / 10000.0};
}
