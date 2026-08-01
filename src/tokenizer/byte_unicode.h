#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

// intermediary between byte and utf
const std::array<uint32_t, 256>& byte_to_codepoint();

// bytes to utf
std::string encode_bytes(std::string_view raw); 
// utf8 back to original bytes
std::string decode_bytes(std::string_view mapped); 
