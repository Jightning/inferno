#include <cstring>

#include "check.h"
#include "tokenizer/byte_unicode.h"

namespace {

// Byte (by list index) to codepoint
constexpr std::array<uint32_t, 256> build_table() {
    std::array<uint32_t, 256> table {};

    auto is_kept = [](uint32_t i) { // chars in a certain range don't change
        return (0x21 <= i && i <= 0x7E) || (0xA1 <= i && i <= 0xAC) || (0xAE <= i && i <= 0xFF);
    };

    // Like having 2 lists, one starts at 0x21 (kept chars), and the other at 0x100 (everything else)
    uint32_t next = 0x100;
    for (uint32_t i = 0; i < 256; ++i) { 
        table[i] = is_kept(i) ? i : next++;  
    }

    return table;
}

constexpr std::array<uint32_t, 256> kByteToCodepoint = build_table();

// For 1-2 byte utf8
struct Utf8 {
    uint8_t len;
    char bytes[2];
};
static_assert(sizeof(Utf8) == 3, "it seems a padding byte was added"); 

// Codepoint into uft8
constexpr std::array<Utf8, 256> build_utf8_table() {
    std::array<Utf8, 256> table {};

    for (uint32_t i = 0; i < 256; ++i) {
        const uint32_t codepoint = kByteToCodepoint[i];
        if (codepoint <= 0x7F) {
            table[i] = Utf8{1, {static_cast<char>(codepoint), 0}};
        } else {  // nothing exceeds 0x143, so two bytes max
            // splits into 2 bytes (first prefix with 110, second prefix with 10) 
            table[i] = Utf8{2, {static_cast<char>(0xC0 | (codepoint >> 6)), 
                                static_cast<char>(0x80 | (codepoint & 0x3F))}}; 
        }
    }

    return table;
}

constexpr std::array<Utf8, 256> kByteToUtf8 = build_utf8_table(); 

// codepoint to byte
constexpr uint32_t kMaxCodepoint = 0x143;

constexpr std::array<int16_t, kMaxCodepoint + 1> build_reverse_table() {
    std::array<int16_t, kMaxCodepoint + 1> table {};

    for (int16_t& entry : table) entry = -1;
    for (uint32_t i = 0; i < 256; ++i) table[kByteToCodepoint[i]] = static_cast<int16_t>(i); 

    return table;
}

constexpr std::array<int16_t, kMaxCodepoint + 1> kCodepointToByte = build_reverse_table();

}  // namespace

const std::array<uint32_t, 256>& byte_to_codepoint() { // codepoint is the chars # in the unicode catalogue
    return kByteToCodepoint;
}

std::string encode_bytes(std::string_view raw) {
    std::string mapped(raw.size() * 2, '\0'); // Utf8 has a max of 2 bytes hence the *2
    char* out = mapped.data();

    for (const unsigned char b : raw) {
        const Utf8& e = kByteToUtf8[b];
        std::memcpy(out, e.bytes, 2);
        out += e.len;
    }

    mapped.resize(static_cast<size_t>(out - mapped.data())); 
    return mapped;
}

std::string decode_bytes(std::string_view mapped) {
    std::string raw(mapped.size(), '\0');
    char* out = raw.data();

    for (size_t i = 0; i < mapped.size(); ) {
        const unsigned char first = static_cast<unsigned char>(mapped[i]);
        uint32_t codepoint = 0;

        if (first < 0x80) { // just one byte
            codepoint = first;
            i += 1;
        } else if ((first & 0xE0) == 0xC0) { // starts with 110 (for the 2 byte thing)
            INFERNO_CHECK(i + 1 < mapped.size(), "decode_bytes: byte should be 2 bytes but exceeds the size of mapped", i);

            const unsigned char second = static_cast<unsigned char>(mapped[i + 1]);            
            INFERNO_CHECK((second & 0xC0) == 0x80, "decode_bytes: wrong byte format for second byte"); // second byte starts with 10
             
            codepoint = static_cast<uint32_t>(((first & 0x1F) << 6) | (second & 0x3F)); 
            i += 2;
        } else {
            INFERNO_CHECK(false, "decode_bytes: wrong byte format");
        }

        INFERNO_CHECK(codepoint <= kMaxCodepoint && kCodepointToByte[codepoint] >= 0, "decode_bytes: codepoint U+{:04X} is not in the byte table", codepoint);
                      
        *out = static_cast<char>(kCodepointToByte[codepoint]);
        out++;
    }

    raw.resize(static_cast<size_t>(out - raw.data())); 
    return raw;
}
