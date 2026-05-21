#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "../kernels/native_kernel.cuh"
#include "../kernels/schoolbook_kernel.cuh"
#include "../kernels/vedic_kernel.cuh"
#include "pairs_io.hpp"

static std::string pairs_path = "../../pairs.json";

static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--input pairs.json]\n";
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            pairs_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::vector<PairRecord> pairs;
    try {
        pairs = load_pairs(pairs_path);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    int passed = 0;
    int total = static_cast<int>(pairs.size());

    for (const auto& p : pairs) {
        long long expected = p.operand_a * p.operand_b;
        long long vedic_result = 0;
        long long school_result = 0;
        float ms = 0.f;

        int n = host_digits::padded_width(p.operand_a, p.operand_b);

        err = vedic_gpu::vedic_multiply_gpu(
            p.operand_a, p.operand_b, &vedic_result, &ms, n);
        if (err != cudaSuccess) {
            std::cerr << "vedic CUDA error: " << cudaGetErrorString(err) << "\n";
            return 1;
        }

        err = schoolbook_gpu::schoolbook_multiply_gpu(
            p.operand_a, p.operand_b, &school_result, &ms, n);
        if (err != cudaSuccess) {
            std::cerr << "schoolbook CUDA error: " << cudaGetErrorString(err) << "\n";
            return 1;
        }

        bool ok = (vedic_result == expected) && (school_result == expected);
        std::cout << "PAIR " << p.operand_a << "x" << p.operand_b
                  << " vedic=" << vedic_result << " school=" << school_result
                  << " expected=" << expected << " "
                  << (ok ? "[PASS]" : "[FAIL]") << "\n";

        if (!ok) {
            std::cerr << "Validation failed at digit_width=" << p.digit_width
                      << " pair_index=" << p.pair_index << "\n";
            return 1;
        }
        passed += 1;
    }

    std::cout << passed << "/" << total << " pairs passed validation.\n";
    return passed == total ? 0 : 1;
}
