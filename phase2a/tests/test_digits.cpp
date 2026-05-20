#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../src/digits.hpp"

int main() {
    const long long roundtrip[] = {0, 1, 9, 10, 99, 100, 9999999LL, 1234567890LL};
    for (long long value : roundtrip) {
        assert(digits::digits_to_int(digits::int_to_digits(value)) == value);
    }

    OperationCounter ctr;
    int product = digits::single_digit_mult(3, 7, &ctr);
    assert(product == 21);
    assert(ctr.multiplications == 1);

    // Verify null counter path does not crash and returns correct product
    int null_product = digits::single_digit_mult(3, 4, nullptr);
    assert(null_product == 12);

    // Verify add_digit_at null counter path does not crash
    std::vector<int> null_test = {5};
    digits::add_digit_at(null_test, 0, 7, nullptr);
    assert(null_test[0] == 2);
    assert(null_test[1] == 1);

    std::vector<int> digits_list = {5};
    OperationCounter add_ctr;
    digits::add_digit_at(digits_list, 0, 7, &add_ctr);
    assert(digits_list.size() >= 2);
    assert(digits_list[0] == 2);
    assert(digits_list[1] == 1);
    assert(add_ctr.additions >= 1);
    assert(add_ctr.carry_propagations >= 1);

    std::cout << "All digit tests passed\n";
    return 0;
}
