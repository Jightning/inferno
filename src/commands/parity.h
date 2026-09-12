#pragma once

#include <span>

namespace parity {

// inferno parity --model <dir> [--data] [--prompts] [--steps] [--tolerance] [--max-seq]
// Returns false if any prompt failed
bool run(int argc, char** argv);

// max |a[i] - b[i]|
float max_abs_diff(std::span<const float> a, std::span<const float> b);

}  // namespace parity
