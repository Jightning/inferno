#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tokenizer/bpe.h"

namespace {

using Pieces = std::vector<std::string>;

// merges, in rank order -- exactly how tokenizer.json's array is read.
MergeTable table_of(const std::vector<std::pair<std::string, std::string>>& merges) {
    MergeTable table;
    for (size_t i = 0; i < merges.size(); ++i) {
        table.add(merges[i].first, merges[i].second, static_cast<int>(i));
    }
    return table;
}

}  // namespace

TEST_CASE("rank lookup") {
    const MergeTable table = table_of({{"l", "o"}, {"lo", "w"}});

    CHECK(table.size() == 2);
    CHECK(table.rank("l", "o") == 0);
    CHECK(table.rank("lo", "w") == 1);

    SUBCASE("an unlearned pair must not merge") {
        CHECK(!table.rank("o", "w").has_value());
        CHECK(!table.rank("w", "lo").has_value());  // order matters
    }
}

TEST_CASE("merges compose left to right when the ranks say so") {
    const MergeTable table = table_of({{"l", "o"}, {"lo", "w"}});
    CHECK(table.apply("low") == Pieces{"low"});
}

TEST_CASE("the LOWEST RANK wins, not the leftmost pair") {
    // The bug this pins: merging left to right gives {"ab", "c"}, which is plausible and
    // wrong. "b c" is rank 0, so it must merge first, and "a bc" was never learned.
    const MergeTable table = table_of({{"b", "c"}, {"a", "b"}});
    CHECK(table.apply("abc") == Pieces{"a", "bc"});
}

TEST_CASE("merging stops when no adjacent pair is in the table") {
    const MergeTable table = table_of({{"x", "y"}});
    CHECK(table.apply("abc") == Pieces{"a", "b", "c"});
}

TEST_CASE("an empty table leaves single characters") {
    const MergeTable table;
    CHECK(table.apply("hi") == Pieces{"h", "i"});
    CHECK(table.apply("").empty());
}

TEST_CASE("multi-byte characters are one symbol, not two") {
    // After the byte<->unicode map, a space is the 2-byte UTF-8 for U+0120. Splitting on
    // bytes instead of characters would produce garbage pieces that match nothing.
    const MergeTable table = table_of({{"Ġ", "a"}});
    CHECK(table.apply("Ġa") == Pieces{"Ġa"});
    CHECK(table.apply("Ġ") == Pieces{"Ġ"});
}

TEST_CASE("a duplicate merge is a parse bug, not a tie") {
    MergeTable table;
    table.add("a", "b", 0);
    CHECK_THROWS_AS(table.add("a", "b", 7), std::runtime_error);
}
