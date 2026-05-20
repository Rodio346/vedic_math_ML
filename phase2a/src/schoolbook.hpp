#pragma once
// Schoolbook partial-products multiplication (ported from schoolbook.py).

#include <utility>
#include <vector>

#include "digits.hpp"

namespace schoolbook {

inline std::pair<long long, OperationCounter> multiply(
    long long a,
    long long b,
    OperationCounter* counter = nullptr) {
    OperationCounter local;
    OperationCounter* ctr = counter != nullptr ? counter : &local;

    if (a < 0 || b < 0) {
        return {0, *ctr};
    }
    if (a == 0 || b == 0) {
        return {0, *ctr};
    }

    std::vector<int> da = digits::int_to_digits(a);
    std::vector<int> db = digits::int_to_digits(b);
    std::vector<int> accumulator = {0};

    for (size_t j = 0; j < db.size(); ++j) {
        int bj = db[j];
        std::vector<int> partial;
        int carry = 0;
        for (size_t i = 0; i < da.size(); ++i) {
            int product = digits::single_digit_mult(da[i], bj, ctr);
            int total = product + carry;
            // NOTE: addition is only recorded when carry != 0. The product+0 case
            // when carry==0 is not counted. This matches the Python Phase 1 behaviour
            // exactly — both phases undercount schoolbook additions by the same amount,
            // so cross-phase comparability is preserved. Absolute addition counts are
            // a lower bound, not an exact total. Do not fix this without updating Phase 1.
            if (carry && ctr != nullptr) {
                ctr->add();
            }
            int partial_digit = total % 10;
            carry = total / 10;
            if (carry && ctr != nullptr) {
                ctr->carry();
            }
            partial.push_back(partial_digit);
        }
        if (carry) {
            partial.push_back(carry);
        }

        for (size_t idx = 0; idx < partial.size(); ++idx) {
            digits::add_digit_at(accumulator, static_cast<int>(j + idx), partial[idx], ctr);
        }
    }

    return {digits::digits_to_int(accumulator), *ctr};
}

inline long long multiply_fast(long long a, long long b) {
    return multiply(a, b, nullptr).first;
}

}  // namespace schoolbook
