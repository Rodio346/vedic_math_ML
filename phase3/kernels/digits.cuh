#pragma once
// Digit utilities matching Phase 1 / Phase 2A (LSB at index 0).

#include <cuda_runtime.h>
#include <vector>

#define MAX_N 9
#define MAX_RESULT_DIGS (2 * MAX_N)
#define MAX_CARRY_BUF MAX_N
#define MAX_COLUMN_DIGS (MAX_N + MAX_CARRY_BUF + 4)

namespace device_digits {

__constant__ int MULT_TABLE[10][10] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18},
    {0, 3, 6, 9, 12, 15, 18, 21, 24, 27},
    {0, 4, 8, 12, 16, 20, 24, 28, 32, 36},
    {0, 5, 10, 15, 20, 25, 30, 35, 40, 45},
    {0, 6, 12, 18, 24, 30, 36, 42, 48, 54},
    {0, 7, 14, 21, 28, 35, 42, 49, 56, 63},
    {0, 8, 16, 24, 32, 40, 48, 56, 64, 72},
    {0, 9, 18, 27, 36, 45, 54, 63, 72, 81},
};

__device__ inline void ensure_len(int* digits, int& len, int need) {
    while (len < need) {
        digits[len++] = 0;
    }
}

__device__ inline void add_digit_at(int* digits, int& len, int position, int digit) {
    if (digit == 0) {
        return;
    }
    ensure_len(digits, len, position + 1);
    int carry = digit;
    int pos = position;
    while (carry) {
        ensure_len(digits, len, pos + 1);
        int total = digits[pos] + carry;
        digits[pos] = total % 10;
        carry = total / 10;
        pos += 1;
    }
}

__device__ inline void add_value_at_position(
    int* digits,
    int& len,
    int position,
    int value) {
    if (value == 0) {
        return;
    }
    int place = position;
    int remaining = value;
    while (remaining) {
        int d = remaining % 10;
        remaining /= 10;
        if (d) {
            add_digit_at(digits, len, place, d);
        }
        place += 1;
    }
}

__device__ inline long long digits_to_int(const int* digits, int len) {
    long long total = 0;
    long long place = 1;
    for (int i = 0; i < len; ++i) {
        total += static_cast<long long>(digits[i]) * place;
        place *= 10;
    }
    return total;
}

__device__ inline int column_products(int k, int n) {
    int left = k + 1;
    int right = 2 * n - 1 - k;
    return left < right ? left : right;
}

}  // namespace device_digits

namespace host_digits {

inline const int MULT_TABLE[10][10] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18},
    {0, 3, 6, 9, 12, 15, 18, 21, 24, 27},
    {0, 4, 8, 12, 16, 20, 24, 28, 32, 36},
    {0, 5, 10, 15, 20, 25, 30, 35, 40, 45},
    {0, 6, 12, 18, 24, 30, 36, 42, 48, 54},
    {0, 7, 14, 21, 28, 35, 42, 49, 56, 63},
    {0, 8, 16, 24, 32, 40, 48, 56, 64, 72},
    {0, 9, 18, 27, 36, 45, 54, 63, 72, 81},
};

inline std::vector<int> int_to_digits(long long value) {
    if (value <= 0) {
        return {0};
    }
    std::vector<int> result;
    while (value > 0) {
        result.push_back(static_cast<int>(value % 10));
        value /= 10;
    }
    return result;
}

inline std::vector<int> pad_to_length(std::vector<int> digits, int length) {
    if (static_cast<int>(digits.size()) >= length) {
        return digits;
    }
    digits.insert(digits.end(), static_cast<size_t>(length - digits.size()), 0);
    return digits;
}

inline int padded_width(long long a, long long b) {
    int na = static_cast<int>(int_to_digits(a).size());
    int nb = static_cast<int>(int_to_digits(b).size());
    return na > nb ? na : nb;
}

inline long long digits_to_int(const std::vector<int>& digits) {
    long long total = 0;
    long long place = 1;
    for (int d : digits) {
        total += d * place;
        place *= 10;
    }
    return total;
}

inline void ensure_len(std::vector<int>& digits, int length) {
    if (static_cast<int>(digits.size()) < length) {
        digits.resize(static_cast<size_t>(length), 0);
    }
}

inline void add_digit_at(std::vector<int>& digits, int position, int digit) {
    if (digit == 0) {
        return;
    }
    ensure_len(digits, position + 1);
    int carry = digit;
    int pos = position;
    while (carry) {
        ensure_len(digits, pos + 1);
        int total = digits[static_cast<size_t>(pos)] + carry;
        digits[static_cast<size_t>(pos)] = total % 10;
        carry = total / 10;
        pos += 1;
    }
}

inline void add_value_at_position(std::vector<int>& digits, int position, int value) {
    if (value == 0) {
        return;
    }
    int place = position;
    int remaining = value;
    while (remaining) {
        int d = remaining % 10;
        remaining /= 10;
        if (d) {
            add_digit_at(digits, place, d);
        }
        place += 1;
    }
}

}  // namespace host_digits
