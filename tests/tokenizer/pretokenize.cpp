#include <doctest/doctest.h>

#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/pretokenize.h"

namespace {

std::vector<std::string> chunks_of(std::string_view text) {
    std::vector<std::string> out;
    for (const std::string_view chunk : pretokenize(text)) out.emplace_back(chunk);
    return out;
}

using Chunks = std::vector<std::string>;

}  // namespace

TEST_CASE("the chunks tile the input exactly") {
    const std::string text = "Q: What is 17 multiplied by 23?\nA:";
    std::string rejoined;
    for (const std::string& chunk : chunks_of(text)) {
        CHECK(!chunk.empty());  // a zero-length chunk means an infinite loop
        rejoined += chunk;
    }
    CHECK(rejoined == text);
}

TEST_CASE("whitespace attaches to the FOLLOWING word") {
    // Alternative B's optional leading character. This one rule is why " capital" is a
    // single token and why the vocab is full of pieces beginning with a space.
    CHECK(chunks_of("The capital of France is")
          == Chunks{"The", " capital", " of", " France", " is"});
}

TEST_CASE("a whitespace run gives back its last character") {
    // Alternative F/G. The four spaces before `print` split 3 + 1: the last space goes
    // to " print" via alternative B.
    CHECK(chunks_of("for i in range(10):\n    print(")
          == Chunks{"for", " i", " in", " range", "(", "1", "0", "):\n", "   ", " print", "("});

    SUBCASE("unless the run ends the string, where there is nothing to give it to") {
        CHECK(chunks_of("  trailing test   ") == Chunks{" ", " trailing", " test", "   "});
    }
}

TEST_CASE("digits are one chunk each") {
    // Alternative C is `\p{N}`, a SINGLE digit -- not the `\p{N}{1,3}` of GPT-4's regex
    // that docs/milestones.md mentions.
    CHECK(chunks_of("Q: What is 17 multiplied by 23?\nA:")
          == Chunks{"Q", ":", " What", " is", " ", "1", "7", " multiplied", " by", " ",
                    "2", "3", "?\n", "A", ":"});
}

TEST_CASE("newline runs and trailing newlines") {
    // Alternative E swallows the whitespace up to and including the last newline;
    // alternative D lets a punctuation run carry its trailing newlines ("):\n").
    CHECK(chunks_of("import numpy as np\n\ndef softmax(x):")
          == Chunks{"import", " numpy", " as", " np", "\n\n", "def", " softmax", "(x", "):"});
}

TEST_CASE("contractions and punctuation runs") {
    SUBCASE("alternative A, case-insensitive") {
        CHECK(chunks_of("don't") == Chunks{"don", "'t"});
        CHECK(chunks_of("DON'T") == Chunks{"DON", "'T"});
        CHECK(chunks_of("they've") == Chunks{"they", "'ve"});
    }
    SUBCASE("a lone apostrophe is not a contraction") {
        CHECK(chunks_of("Translate to German: 'Good morning, how are you?'")
              == Chunks{"Translate", " to", " German", ":", " '", "Good", " morning", ",",
                        " how", " are", " you", "?'"});
    }
}

TEST_CASE("degenerate inputs") {
    CHECK(chunks_of("").empty());
    CHECK(chunks_of(" ") == Chunks{" "});
    CHECK(chunks_of("\n") == Chunks{"\n"});
    CHECK(chunks_of("A") == Chunks{"A"});
}

TEST_CASE("next_chunk_length never stalls") {
    // Whatever the input, alternative G is total: every position consumes at least one
    // byte. If this ever fails, pretokenize hangs instead of failing a test.
    const std::string text = "a1 !\n\t\r\n  z'sZ";
    for (size_t i = 0; i < text.size(); ++i) {
        const size_t len = next_chunk_length(text, i);
        CHECK(len >= 1);
        CHECK(i + len <= text.size());
    }
}
