#include <doctest/doctest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "tokenizer/byte_unicode.h"

TEST_CASE("byte_to_codepoint is the GPT-2 table") {
    const auto& table = byte_to_codepoint();

    SUBCASE("printable ASCII maps to itself") {
        CHECK(table['A'] == 0x41);
        CHECK(table['!'] == 0x21);
        CHECK(table['~'] == 0x7E);
    }

    SUBCASE("the 68 unprintable bytes land in U+0100..U+0143, in byte order") {
        CHECK(table[0x00] == 0x100);
        CHECK(table[0x0A] == 0x10A);  // newline -> the char you will see as 'C-dot'
        CHECK(table[0x20] == 0x120);  // space   -> the 'G-dot' in every debug dump
        CHECK(table[0x7F] == 0x121);  // 0x7F picks up right after 0x20's block
        CHECK(table[0xA0] == 0x142);
        CHECK(table[0xAD] == 0x143);  // the last one: soft hyphen
    }

    SUBCASE("the high latin-1 range maps to itself") {
        CHECK(table[0xA1] == 0xA1);
        CHECK(table[0xAE] == 0xAE);
        CHECK(table[0xFF] == 0xFF);
    }

    SUBCASE("it is a bijection: 256 distinct codepoints") {
        std::vector<uint32_t> seen(table.begin(), table.end());
        std::sort(seen.begin(), seen.end());
        CHECK(std::unique(seen.begin(), seen.end()) == seen.end());
    }
}

TEST_CASE("encode_bytes produces the strings the vocab is written in") {
    CHECK(encode_bytes("The") == "The");
    CHECK(encode_bytes(" capital") == "Ġcapital");
    CHECK(encode_bytes("\n\n") == "ĊĊ");
    CHECK(encode_bytes("") == "");
}

TEST_CASE("decode_bytes inverts encode_bytes") {
    SUBCASE("every one of the 256 byte values round-trips") {
        std::string all;
        for (int b = 0; b < 256; ++b) all.push_back(static_cast<char>(b));
        CHECK(decode_bytes(encode_bytes(all)) == all);
    }

    SUBCASE("UTF-8 input survives, byte by byte") {
        const std::string text = "café — 日本語";
        CHECK(decode_bytes(encode_bytes(text)) == text);
    }

    SUBCASE("a codepoint outside the table is refused, not guessed") {
        CHECK_THROWS_AS(decode_bytes("☃"), std::runtime_error);  // snowman
    }
}
