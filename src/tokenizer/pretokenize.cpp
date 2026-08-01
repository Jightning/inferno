#include <array>
#include <cstdint>

#include "tokenizer/pretokenize.h"

namespace {

enum : uint8_t { LETTER = 1, DIGIT = 2, SPACE = 4, NEWLINE = 8 };

// For checking the character type (letter, digit, space, or newline)
constexpr std::array<uint8_t, 256> build_class_table() {
    std::array<uint8_t, 256> table {};

    for (int c = 'a'; c <= 'z'; ++c) table[c] = LETTER;
    for (int c = 'A'; c <= 'Z'; ++c) table[c] = LETTER; 
    for (int c = '0'; c <= '9'; ++c) table[c] = DIGIT;

    table[' '] = table['\t'] = table['\v'] = table['\f'] = SPACE; 
    table['\r'] = table['\n'] = SPACE | NEWLINE;

    for (int c = 0x80; c < 256; ++c) table[c] = LETTER;

    return table;
}

constexpr std::array<uint8_t, 256> kClass = build_class_table();

// helpers
constexpr bool is(uint8_t ch, uint8_t flags) { return (kClass[ch] & flags) != 0; }
constexpr uint8_t lower(uint8_t ch) { return (ch >= 'A' && ch <= 'Z') ? ch + 32 : ch; }

}  // namespace

size_t next_chunk_length(std::string_view text, size_t pos) {
    const size_t n = text.size();
    // peak syntax with is(at, enum)
    const auto at = [&text](size_t k) { return static_cast<uint8_t>(text[k]); }; 

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    // contractions/suffixes like 's, 't, 're etc.
    if (at(pos) == '\'' && pos + 1 < n) {
        const uint8_t a = lower(at(pos + 1)); 
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (pos + 2 < n) {
            const uint8_t b = lower(at(pos + 2));
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    // non-letter/non-number char followed by letters
    size_t i = pos;
    if (!is(at(i), LETTER | DIGIT | NEWLINE)) ++i; 
    if (i < n && is(at(i), LETTER)) {
        while (i < n && is(at(i), LETTER)) ++i;
        return i - pos;
    }

    // \p{N}
    // a digit
    if (is(at(pos), DIGIT)) return 1;

    // ?[^\s\p{L}\p{N}]+[\r\n]*
    // space, punctuation, trailing newlines
    i = pos; 
    if (at(i) == ' ') ++i;
    if (i < n && !is(at(i), SPACE | LETTER | DIGIT)) {
        while (i < n && !is(at(i), SPACE | LETTER | DIGIT)) ++i;
        while (i < n && is(at(i), NEWLINE)) ++i;
        return i - pos; 
    }

    // go through all the following spaces until a non-whitespace is found (end)
    size_t end = pos;
    size_t last_newline = n;
    while (end < n && is(at(end), SPACE)) {
        if (is(at(end), NEWLINE)) {
            last_newline = end;
        }

        ++end;
    }

    // pos isn't a whitespace
    if (end == pos) return 1;

    // \s*[\r\n]+
    // whitespace followed by newline
    if (last_newline != n) return last_newline + 1 - pos;

    // \s+(?!\S)
    // whitespace followed by a space
    if (end == n) return end - pos;
    if (end - pos >= 2) return end - pos - 1;

    // \s+
    // any whitespace
    return 1; 
}

std::vector<std::string_view> pretokenize(std::string_view text) {
    std::vector<std::string_view> chunks; 
    chunks.reserve(text.size() / 4 + 1);  // ~4 bytes per chunk 

    for (size_t pos = 0; pos < text.size(); ) {
        const size_t len = next_chunk_length(text, pos);
        chunks.push_back(text.substr(pos, len)); 
        pos += len;
    }

    return chunks;
}
