// Need /model files, skipped if missing

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include "loader/npy.h"
#include "tokenizer/tokenizer.h"

#ifdef INFERNO_MODEL_DIR

namespace {

std::filesystem::path tokenizer_json() {
    return std::filesystem::path(INFERNO_MODEL_DIR) / "tokenizer.json";
}

const Tokenizer& shared_tokenizer() {
    static const Tokenizer tok(tokenizer_json().string());
    return tok;
}

bool model_available() {
    if (std::filesystem::exists(tokenizer_json())) return true;
    MESSAGE("skipped: no tokenizer.json at " << tokenizer_json().string());
    return false;
}

}  // namespace

TEST_CASE("the tables load with the counts tokenizer.json actually holds") {
    if (!model_available()) return;
    const Tokenizer& tok = shared_tokenizer();

    // 151643 vocab entries (ids 0..151642) + 22 added tokens (151643..151664).
    // Well below ModelConfig::vocab_size (151936): the embedding matrix is padded.
    CHECK(tok.size() == 151643 + 22);

    SUBCASE("pieces resolve to the ids Hugging Face uses") {
        CHECK(tok.piece_to_id("The") == 785);
        CHECK(tok.piece_to_id("Ġcapital") == 6722);   // note the byte-mapped space
        CHECK(tok.piece_to_id("<|im_end|>") == 151645);
        CHECK(!tok.piece_to_id("Ġcapital_no_such_piece").has_value());
    }

    SUBCASE("and back") {
        CHECK(tok.id_to_piece(785) == "The");
        CHECK(tok.id_to_piece(151645) == "<|im_end|>");
        CHECK_THROWS_AS(tok.id_to_piece(151936), std::runtime_error);
        CHECK_THROWS_AS(tok.id_to_piece(-1), std::runtime_error);
    }
}

TEST_CASE("a malformed tokenizer.json is rejected, loudly") {
    CHECK_THROWS_AS(Tokenizer("/nonexistent/inferno/tokenizer.json"), std::runtime_error);
}

TEST_CASE("encode matches Hugging Face on the shapes that trip tokenizers") {
    if (!model_available()) return;
    const Tokenizer& tok = shared_tokenizer();

    SUBCASE("no BOS, no EOS -- chat_template is false") {
        CHECK(tok.encode("The capital of France is")
              == std::vector<int>{785, 6722, 315, 9625, 374});
    }
    SUBCASE("digits split one by one") {
        CHECK(tok.encode("Q: What is 17 multiplied by 23?\nA:")
              == std::vector<int>{48, 25, 3555, 374, 220, 16, 22, 54916, 553, 220, 17, 18,
                                  5267, 32, 25});
    }
    SUBCASE("code, newlines and indentation") {
        CHECK(tok.encode("for i in range(10):\n    print(")
              == std::vector<int>{1958, 600, 304, 2088, 7, 16, 15, 982, 262, 1173, 7});
    }
    SUBCASE("added tokens are matched before BPE ever runs") {
        CHECK(tok.encode("a<|im_end|>b") == std::vector<int>{64, 151645, 65});
    }
    SUBCASE("empty in, empty out") {
        CHECK(tok.encode("").empty());
    }
}

TEST_CASE("decode inverts encode") {
    if (!model_available()) return;
    const Tokenizer& tok = shared_tokenizer();

    for (const std::string& text : {std::string("The capital of France is"),
                                    std::string("for i in range(10):\n    print("),
                                    std::string("  leading and trailing   "),
                                    std::string("café — 日本語"),
                                    std::string("")}) {
        CAPTURE(text);
        CHECK(tok.decode(tok.encode(text)) == text);
    }

    SUBCASE("an id with no piece is refused, not silently dropped") {
        const std::vector<int> bad{785, 999999};
        CHECK_THROWS_AS(tok.decode(bad), std::runtime_error);
    }
}

#ifdef PARITY_DATA_DIR
TEST_CASE("THE GATE: all 20 parity prompts encode id for id") {
    if (!model_available()) return;

    const std::filesystem::path dir = PARITY_DATA_DIR;
    const std::filesystem::path manifest_path = dir / "manifest.json";
    if (!std::filesystem::exists(manifest_path)) {
        MESSAGE("skipped: no parity data at " << dir.string());
        return;
    }

    std::ifstream manifest_file(manifest_path);
    const nlohmann::json manifest = nlohmann::json::parse(manifest_file);
    const auto prompts = manifest.at("prompts").get<std::vector<std::string>>();
    const auto n_generated = manifest.at("n_tokens").get<size_t>();
    REQUIRE(prompts.size() == 20);

    const Tokenizer& tok = shared_tokenizer();
    size_t matched = 0;

    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::filesystem::path tokens_path = dir / std::format("prompt{:02}_tokens.npy", i);
        if (!std::filesystem::exists(tokens_path)) continue;

        CAPTURE(i);
        CAPTURE(prompts[i]);

        const std::vector<int64_t> expected = load_npy_i64(tokens_path.string());
        const std::vector<int> actual = tok.encode(prompts[i]);

        // The length check is half the test: an encoding that is one token short would
        // still match the prefix. The file is prompt + n_tokens generated.
        CHECK(actual.size() + n_generated == expected.size());

        bool same = actual.size() + n_generated == expected.size();
        for (size_t t = 0; same && t < actual.size(); ++t) {
            if (expected[t] != actual[t]) {

                MESSAGE("prompt " << i << " diverges at token " << t << ": expected "
                                  << expected[t] << " got " << actual[t]);
                same = false;
            }
        }
        CHECK(same);
        if (same) ++matched;
    }

    CHECK(matched == 20);
}
#endif  // PARITY_DATA_DIR

#endif  // INFERNO_MODEL_DIR
