#include <cassert>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "../src/digits.hpp"
#include "../src/schoolbook.hpp"
#include "../src/vedic.hpp"

struct Case {
    long long a;
    long long b;
    long long expected;
    bool check_mults;
};

static int digit_width(long long value) {
    return static_cast<int>(digits::int_to_digits(value).size());
}

static int max_width(long long a, long long b) {
    int wa = digit_width(a);
    int wb = digit_width(b);
    return wa > wb ? wa : wb;
}

int main() {
    const Case cases[] = {
        {23, 41, 943, true},
        {99, 99, 9801, true},
        {100, 100, 10000, true},
        {999, 999, 998001, true},
        {0, 5, 0, true},
        {1, 1, 1, true},
        {10, 10, 100, true},
        {12, 34, 408, true},
        {56, 78, 4368, true},
        {12345, 67890, 838102050LL, true},
        {9999999LL, 9999999LL, 99999980000001LL, true},
    };

    for (const Case& c : cases) {
        OperationCounter v_ctr, s_ctr;
        long long v = vedic::multiply(c.a, c.b, &v_ctr).first;
        long long s = schoolbook::multiply(c.a, c.b, &s_ctr).first;
        long long expected = c.a * c.b;

        if (v != s || v != expected) {
            std::cerr << "FAIL " << c.a << "x" << c.b
                      << " v=" << v << " s=" << s << " expected=" << expected << "\n";
            std::abort();
        }

        if (c.check_mults && c.a != 0 && c.b != 0) {
            int n = max_width(c.a, c.b);
            int n_sq = n * n;
            if (v_ctr.multiplications != n_sq) {
                std::cerr << "FAIL mults vedic " << c.a << "x" << c.b << "\n";
                std::abort();
            }
            int da_len = static_cast<int>(digits::int_to_digits(c.a).size());
            int db_len = static_cast<int>(digits::int_to_digits(c.b).size());
            if (s_ctr.multiplications != da_len * db_len) {
                std::cerr << "FAIL mults schoolbook " << c.a << "x" << c.b << "\n";
                std::abort();
            }
        }

        std::cout << "PASS " << c.a << "x" << c.b << "\n";
    }

    std::cout << "All algorithm tests passed\n";
    return 0;
}
