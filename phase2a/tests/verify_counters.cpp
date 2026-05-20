#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "../src/schoolbook.hpp"
#include "../src/vedic.hpp"

int main() {
    std::ifstream in("verification/counter_parity.json");
    if (!in) {
        in.open("../verification/counter_parity.json");
    }
    if (!in) {
        std::cerr << "Cannot open verification/counter_parity.json\n";
        return 1;
    }

    nlohmann::json doc;
    in >> doc;

    int passed = 0;
    int total = 0;

    for (const auto& entry : doc["entries"]) {
        long long a = entry["operand_a"].get<long long>();
        long long b = entry["operand_b"].get<long long>();
        std::string method = entry["method"].get<std::string>();
        int exp_m = entry["multiplications"].get<int>();
        int exp_a = entry["additions"].get<int>();
        int exp_c = entry["carry_propagations"].get<int>();

        OperationCounter ctr;
        if (method == "vedic") {
            vedic::multiply(a, b, &ctr);
        } else {
            schoolbook::multiply(a, b, &ctr);
        }

        bool ok = ctr.multiplications == exp_m && ctr.additions == exp_a &&
                  ctr.carry_propagations == exp_c;
        total += 1;
        if (ok) {
            passed += 1;
        }

        std::cout << "PAIR " << a << "x" << b << " " << method << ": mults=" << ctr.multiplications
                  << " adds=" << ctr.additions << " carries=" << ctr.carry_propagations
                  << (ok ? " [PASS]\n" : " [FAIL]\n");
    }

    std::cout << "Summary: " << passed << "/" << total << " passed\n";
    return passed == total ? 0 : 1;
}
