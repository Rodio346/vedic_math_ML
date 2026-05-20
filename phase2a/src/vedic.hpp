#pragma once
// Urdhva-Tiryagbhyam column-wise multiplication (ported from vedic.py).

#include <utility>
#include <vector>

#include "digits.hpp"

namespace vedic {

// Multiply a and b using Urdhva-Tiryagbhyam.
//
// Counter usage: if a non-null counter is passed, it is populated in-place
// via the pointer AND a copy is returned in the pair's second field.
// Always use the passed-in counter (not the return value's counter) when
// accumulating counts across multiple calls, to avoid working with a stale copy.
//
// Example:
//   OperationCounter ctr;
//   auto [result, _] = vedic::multiply(a, b, &ctr);  // use ctr, not _
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

    std::vector<int> da_raw = digits::int_to_digits(a);
    std::vector<int> db_raw = digits::int_to_digits(b);
    int n = static_cast<int>(da_raw.size());
    if (static_cast<int>(db_raw.size()) > n) {
        n = static_cast<int>(db_raw.size());
    }
    std::vector<int> da = digits::pad_to_length(da_raw, n);
    std::vector<int> db = digits::pad_to_length(db_raw, n);

    std::vector<int> result_digits;
    std::vector<int> carry_digits;

    for (int k = 0; k < 2 * n - 1; ++k) {
        std::vector<int> column_digits;

        for (int i = 0; i < n; ++i) {
            int j = k - i;
            if (0 <= j && j < n) {
                int product = digits::single_digit_mult(da[static_cast<size_t>(i)],
                                                        db[static_cast<size_t>(j)],
                                                        ctr);
                digits::add_value_at_position(column_digits, 0, product, ctr);
            }
        }

        for (size_t idx = 0; idx < carry_digits.size(); ++idx) {
            digits::add_digit_at(column_digits, static_cast<int>(idx), carry_digits[idx], ctr);
        }
        carry_digits.clear();

        if (column_digits.empty()) {
            column_digits.push_back(0);
        }

        result_digits.push_back(column_digits[0]);

        if (column_digits.size() > 1) {
            for (size_t i = 1; i < column_digits.size(); ++i) {
                if (column_digits[i] && ctr != nullptr) {
                    ctr->carry();
                }
            }
            carry_digits.assign(column_digits.begin() + 1, column_digits.end());
        }
    }

    for (size_t idx = 0; idx < carry_digits.size(); ++idx) {
        digits::add_digit_at(result_digits, (2 * n - 1) + static_cast<int>(idx),
                             carry_digits[idx], ctr);
    }

    return {digits::digits_to_int(result_digits), *ctr};
}

inline long long multiply_fast(long long a, long long b) {
    return multiply(a, b, nullptr).first;
}

}  // namespace vedic
