#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../kernels/native_kernel.cuh"
#include "../kernels/schoolbook_kernel.cuh"
#include "../kernels/vedic_kernel.cuh"
#include "pairs_io.hpp"

struct GpuInfo {
    std::string name;
    std::string compute_capability;
    int warp_size = 32;
    int shared_mem_per_block = 0;
};

struct TimingStats {
    double mean_us = 0.0;
    double std_us = 0.0;
};

static std::string pairs_path = "../../pairs.json";
static std::string output_path = "results/phase3.csv";
static std::vector<int> digit_widths = {4, 5, 6, 7, 8, 9};
static int iterations = 10000;
static int warmup = 100;

static TimingStats compute_stats(const std::vector<double>& samples_us) {
    TimingStats s;
    if (samples_us.empty()) {
        return s;
    }
    double sum = 0.0;
    for (double v : samples_us) sum += v;
    s.mean_us = sum / static_cast<double>(samples_us.size());
    double var = 0.0;
    for (double v : samples_us) {
        double d = v - s.mean_us;
        var += d * d;
    }
    s.std_us = std::sqrt(var / static_cast<double>(samples_us.size()));
    return s;
}

static GpuInfo query_gpu() {
    GpuInfo info;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    info.name = prop.name;
    std::ostringstream cc;
    cc << prop.major << "." << prop.minor;
    info.compute_capability = cc.str();
    info.warp_size = prop.warpSize;
    info.shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
    return info;
}

static std::string make_run_id() {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << "phase3_" << now;
    return oss.str();
}

static void write_csv_header(std::ofstream& out) {
    out << "run_id,digit_width,pair_index,operand_a,operand_b,method,"
           "threads_per_column_block,mean_time_us,std_time_us,"
           "speedup_vs_single_thread,gpu_device,compute_capability,"
           "warp_size,shared_mem_per_block_bytes\n";
}

static void run_timed(
    long long a,
    long long b,
    const std::string& method,
    int threads,
    int iters,
    int warm,
    std::vector<double>& samples_us,
    long long& out_result) {
    samples_us.clear();
    float ms = 0.f;
    cudaError_t err = cudaSuccess;

    auto run_once = [&]() {
        long long r = 0;
        if (method == "vedic") {
            err = vedic_gpu::vedic_multiply_gpu(a, b, &r, &ms, threads);
        } else if (method == "schoolbook" || method == "schoolbook_sweep") {
            err = schoolbook_gpu::schoolbook_multiply_gpu(a, b, &r, &ms, threads);
        } else if (method == "native") {
            err = native_gpu::native_multiply_gpu_with_digits(a, b, &r, &ms);
        }
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        return static_cast<double>(ms) * 1000.0;
    };

    for (int i = 0; i < warm; ++i) {
        (void)run_once();
    }
    for (int i = 0; i < iters; ++i) {
        samples_us.push_back(run_once());
    }
    out_result = 0;
    err = (method == "native")
        ? native_gpu::native_multiply_gpu_with_digits(a, b, &out_result, &ms)
        : (method == "vedic")
            ? vedic_gpu::vedic_multiply_gpu(a, b, &out_result, &ms, threads)
            : schoolbook_gpu::schoolbook_multiply_gpu(a, b, &out_result, &ms, threads);
    (void)err;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            pairs_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--digit-widths") {
            digit_widths.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                digit_widths.push_back(std::stoi(argv[++i]));
            }
        }
    }

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    GpuInfo gpu = query_gpu();
    std::string run_id = make_run_id();

    std::vector<PairRecord> pairs;
    try {
        pairs = filter_pairs(load_pairs(pairs_path), digit_widths);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "cannot open output: " << output_path << "\n";
        return 1;
    }
    write_csv_header(out);

    for (const auto& p : pairs) {
        int n = host_digits::padded_width(p.operand_a, p.operand_b);
        auto sweep = thread_sweep(n);

        std::map<std::string, std::map<int, double>> mean_by_method_thread;

        const std::vector<std::pair<std::string, bool>> methods = {
            {"vedic", true},
            {"schoolbook", false},
            {"schoolbook_sweep", true},
            {"native", false},
        };

        for (const auto& [method, use_sweep] : methods) {
            std::vector<int> threads_list;
            if (method == "native") {
                threads_list = {1};
            } else if (method == "schoolbook") {
                threads_list = {n};
            } else if (use_sweep) {
                threads_list = sweep;
            }

            for (int t : threads_list) {
                std::vector<double> samples;
                long long result = 0;
                try {
                    run_timed(
                        p.operand_a, p.operand_b, method, t,
                        iterations, warmup, samples, result);
                } catch (const std::exception& ex) {
                    std::cerr << "benchmark error: " << ex.what() << "\n";
                    return 1;
                }

                TimingStats stats = compute_stats(samples);
                mean_by_method_thread[method][t] = stats.mean_us;

                double speedup = 1.0;
                auto it_single = mean_by_method_thread[method].find(1);
                if (it_single != mean_by_method_thread[method].end() && stats.mean_us > 0) {
                    speedup = it_single->second / stats.mean_us;
                }

                out << run_id << "," << p.digit_width << "," << p.pair_index << ","
                    << p.operand_a << "," << p.operand_b << "," << method << ","
                    << t << "," << std::fixed << std::setprecision(3) << stats.mean_us
                    << "," << stats.std_us << "," << std::setprecision(6) << speedup
                    << "," << gpu.name << "," << gpu.compute_capability << ","
                    << gpu.warp_size << "," << gpu.shared_mem_per_block << "\n";
            }
        }
    }

    std::cout << "Wrote " << output_path << " (" << pairs.size() << " pairs)\n";
    return 0;
}
