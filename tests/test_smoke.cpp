#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>

TEST_CASE("test harness runs under sanitizers") {
    std::vector<int> v{1, 2, 3};
    int sum = 0;
    for (int x : v) {
        sum += x;
    }
    CHECK(sum == 6);
}
