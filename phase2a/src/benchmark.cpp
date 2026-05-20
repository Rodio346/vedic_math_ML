// Phase 2A benchmark runner — comparable CSV output to Phase 1 Python.

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "depth.hpp"
#include "native.hpp"
#include "schoolbook.hpp"
#include "vedic.hpp"

namespace {

struct PairEntry {
    int digit_width;
    int pair_index;
    long long operand_a;
    long long operand_b;
};

struct TimingResult {
    std::vector<double> repeat_times_us;
    double mean_us = 0.0;
    double std_us = 0.0;
};

std::string platform_string() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "Unknown";
#endif
}

std::string run_id_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &t);
#else
    gmtime_r(&t, &utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &utc);
    std::ostringstream oss;
    oss << "phase2a-" << buf << "-" << std::hex << (std::rand() & 0xFFFFFF);
    return oss.str();
}

double sample_stdev(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    double mean = 0.0;
    for (double v : values) {
        mean += v;
    }
    mean /= static_cast<double>(values.size());
    double var = 0.0;
    for (double v : values) {
        double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(values.size() - 1);
    return std::sqrt(var);
}

template <typename Fn>
double time_block_us(Fn fn, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    return (seconds / static_cast<double>(iterations)) * 1'000'000.0;
}

TimingResult time_method(
    const std::string& method,
    long long a,
    long long b,
    int warmup,
    int iterations,
    int repeats) {
    auto run_once = [&]() {
        if (method == "vedic") {
            (void)vedic::multiply_fast(a, b);
        } else if (method == "schoolbook") {
            (void)schoolbook::multiply_fast(a, b);
        } else {
            (void)native::multiply_fast(a, b);
        }
    };

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    TimingResult result;
    for (int r = 0; r < repeats; ++r) {
        double us = time_block_us(run_once, iterations);
        result.repeat_times_us.push_back(us);
    }
    double sum = 0.0;
    for (double v : result.repeat_times_us) {
        sum += v;
    }
    result.mean_us = sum / static_cast<double>(result.repeat_times_us.size());
    result.std_us = sample_stdev(result.repeat_times_us);
    return result;
}

void write_csv_row(
    std::ofstream& out,
    const std::string& run_id,
    const PairEntry& pair,
    const std::string& method,
    long long product,
    const TimingResult& timing,
    const OperationCounter& ctr,
    const DepthMetrics& depth,
    int repeats,
    int warmup,
    const std::string& compiler_flags,
    const std::string& platform) {
    out << run_id << ',' << pair.digit_width << ',' << pair.pair_index << ','
        << pair.operand_a << ',' << pair.operand_b << ',' << product << ',' << method;
    for (int i = 0; i < repeats; ++i) {
        if (i < static_cast<int>(timing.repeat_times_us.size())) {
            out << ',' << std::fixed << std::setprecision(3) << timing.repeat_times_us[static_cast<size_t>(i)];
        } else {
            out << ',';
        }
    }
    out << ',' << std::fixed << std::setprecision(3) << timing.mean_us << ','
        << std::fixed << std::setprecision(3) << timing.std_us << ','
        << ctr.multiplications << ',' << ctr.additions << ',' << ctr.carry_propagations << ','
        << ctr.total_ops() << ',';
    if (method == "native") {
        out << "0,0,0.0";
    } else {
        out << depth.sequential_depth << ',' << depth.parallel_width << ','
            << std::fixed << std::setprecision(4) << depth.parallelism_score;
    }
    out << ',' << compiler_flags << ',' << warmup << ',' << platform << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::string input = "../../pairs.json";
    std::string output = "results/phase2a.csv";
    int warmup = 1000;
    int iterations = 100'000;
    int repeats = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        }
    }

    std::ifstream in(input);
    if (!in) {
        std::cerr << "Cannot open input: " << input << "\n";
        return 1;
    }

    nlohmann::json doc;
    in >> doc;

    std::vector<PairEntry> pairs;
    for (const auto& p : doc["pairs"]) {
        pairs.push_back({p["digit_width"].get<int>(), p["pair_index"].get<int>(),
                         p["operand_a"].get<long long>(), p["operand_b"].get<long long>()});
    }

    if (pairs.empty()) {
        std::cerr << "ERROR: No pairs loaded from " << input << "\n"
                  << "Expected JSON schema: {\"pairs\": [{\"digit_width\": N, "
                  << "\"pair_index\": N, \"operand_a\": N, \"operand_b\": N}, ...]}\n"
                  << "Got top-level keys: ";
        for (auto& el : doc.items()) {
            std::cerr << "\"" << el.key() << "\" ";
        }
        std::cerr << "\n";
        return 1;
    }

    const std::string run_id = run_id_now();
    const std::string compiler_flags =
#if defined(_MSC_VER)
        "/O2 /std:c++17";
#else
        "-O2 -std=c++17";
#endif
    const std::string platform = platform_string();

    std::ofstream out(output);
    if (!out) {
        std::cerr << "Cannot open output: " << output << "\n";
        return 1;
    }

    out << "run_id,digit_width,pair_index,operand_a,operand_b,product,method";
    for (int i = 1; i <= repeats; ++i) {
        out << ",time_repeat_" << i;
    }
    out << ",mean_time_us,std_time_us,multiplications,additions,carry_propagations,total_ops,"
           "sequential_depth,parallel_width,parallelism_score,compiler_flags,warmup_iterations,"
           "platform\n";

    const char* methods[] = {"vedic", "schoolbook", "native"};

    for (const PairEntry& pair : pairs) {
        OperationCounter v_ctr, s_ctr;
        long long v = vedic::multiply(pair.operand_a, pair.operand_b, &v_ctr).first;
        long long s = schoolbook::multiply(pair.operand_a, pair.operand_b, &s_ctr).first;
        long long n = native::multiply_fast(pair.operand_a, pair.operand_b);
        long long expected = pair.operand_a * pair.operand_b;

        if (v != s || v != expected || n != expected) {
            std::cerr << "CORRECTNESS FAILURE width=" << pair.digit_width
                      << " pair=" << pair.pair_index << " (" << pair.operand_a << ", "
                      << pair.operand_b << ")\n";
            return 1;
        }

        DepthMetrics v_depth = vedic_depth_metrics(pair.digit_width);
        DepthMetrics s_depth = schoolbook_depth_metrics(pair.digit_width);
        DepthMetrics empty_depth{0, 0, 0, 0.0};

        for (const char* method : methods) {
            TimingResult timing = time_method(method, pair.operand_a, pair.operand_b, warmup,
                                              iterations, repeats);
            OperationCounter ctr;
            DepthMetrics depth = empty_depth;
            if (method == std::string("vedic")) {
                ctr = v_ctr;
                depth = v_depth;
            } else if (method == std::string("schoolbook")) {
                ctr = s_ctr;
                depth = s_depth;
            }
            write_csv_row(out, run_id, pair, method, expected, timing, ctr, depth, repeats,
                          warmup, compiler_flags, platform);
        }
    }

    std::cout << "Wrote " << output << " run_id=" << run_id << '\n';
    return 0;
}
