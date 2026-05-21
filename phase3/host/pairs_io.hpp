#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

struct PairRecord {
    int digit_width = 0;
    int pair_index = 0;
    long long operand_a = 0;
    long long operand_b = 0;
};

inline std::vector<PairRecord> load_pairs(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open pairs file: " + path);
    }
    nlohmann::json data = nlohmann::json::parse(in);
    std::vector<PairRecord> pairs;
    for (const auto& item : data.at("pairs")) {
        PairRecord p;
        p.digit_width = item.at("digit_width").get<int>();
        p.pair_index = item.at("pair_index").get<int>();
        p.operand_a = item.at("operand_a").get<long long>();
        p.operand_b = item.at("operand_b").get<long long>();
        pairs.push_back(p);
    }
    return pairs;
}

inline std::vector<PairRecord> filter_pairs(
    const std::vector<PairRecord>& pairs,
    const std::vector<int>& widths) {
    std::vector<PairRecord> out;
    for (const auto& p : pairs) {
        for (int w : widths) {
            if (p.digit_width == w) {
                out.push_back(p);
                break;
            }
        }
    }
    return out;
}

inline int cap_threads(int t) {
    if (t < 1) return 1;
    if (t > 32) return 32;
    return t;
}

inline std::vector<int> thread_sweep(int n) {
    int vals[] = {
        1,
        (n + 1) / 2,
        n,
        n + 1,
        n + 2,
        cap_threads(2 * n),
        cap_threads(n * n),
    };
    std::vector<int> out;
    for (int v : vals) {
        bool dup = false;
        for (int e : out) {
            if (e == v) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            out.push_back(v);
        }
    }
    return out;
}
