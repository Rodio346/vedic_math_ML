#pragma once
// Digit-list utilities and instrumented single-digit arithmetic (ported from _digits.py).

#include <vector>

struct OperationCounter {
    int multiplications = 0;
    int additions = 0;
    int carry_propagations = 0;

    int total_ops() const {
        return multiplications + additions + carry_propagations;
    }

    void reset() {
        multiplications = 0;
        additions = 0;
        carry_propagations = 0;
    }

    void multiply() { multiplications += 1; }
    void add() { additions += 1; }
    void carry() { carry_propagations += 1; }
};

namespace digits {

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
    if (value < 0) {
        return {0};
    }
    if (value == 0) {
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

inline long long digits_to_int(const std::vector<int>& digits) {
    long long total = 0;
    long long place = 1;
    for (int digit : digits) {
        total += digit * place;
        place *= 10;
    }
    return total;
}

inline int single_digit_mult(int d1, int d2, OperationCounter* counter) {
    if (counter != nullptr) {
        counter->multiply();
    }
    return MULT_TABLE[d1][d2];
}

inline void ensure_len(std::vector<int>& digits, int length) {
    if (static_cast<int>(digits.size()) < length) {
        digits.resize(static_cast<size_t>(length), 0);
    }
}

inline void add_digit_at(
    std::vector<int>& digits,
    int position,
    int digit,
    OperationCounter* counter) {
    ensure_len(digits, position + 1);
    int carry = digit;
    int pos = position;
    while (carry) {
        ensure_len(digits, pos + 1);
        int total = digits[static_cast<size_t>(pos)] + carry;
        if (counter != nullptr) {
            counter->add();
        }
        digits[static_cast<size_t>(pos)] = total % 10;
        carry = total / 10;
        if (carry && counter != nullptr) {
            counter->carry();
        }
        pos += 1;
    }
}

inline void add_value_at_position(
    std::vector<int>& digits,
    int position,
    int value,
    OperationCounter* counter) {
    if (value == 0) {
        return;
    }
    int place = position;
    int remaining = value;
    while (remaining) {
        int d = remaining % 10;
        remaining /= 10;
        if (d) {
            add_digit_at(digits, place, d, counter);
        }
        place += 1;
    }
}

}  // namespace digits
