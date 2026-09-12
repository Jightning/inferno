#pragma once

#include <cstddef>
#include <string_view>

namespace generate {

// inferno generate --model <dir> [--prompt | --token-ids] [--n-tokens] [--max-seq]
void run(int argc, char** argv);

// Provides the amount of s that is "good" (no incomplete characters)
// incomplete is defined by a lead byte showing a byte length that doesn't match the current length
size_t complete_utf8_prefix_len(std::string_view s);

}  // namespace generate
