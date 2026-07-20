#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

#include "safetensors.h"

class Weights {
    // Owns the numbers: one fp32 vector per tensor, converted from BF16 in the
    // constructor. After conversion the mmap is no longer needed — Weights owns
    // its own data.
    std::unordered_map<std::string, std::vector<float>> data_;
public:
    // Takes ownership of the parsed file; validates every shape against ref §18's
    // table (INFERNO_CHECK, naming the tensor) BEFORE converting, then converts.
    Weights(SafeTensorModel&& st, const ModelConfig& cfg);

    // Generic path: quantizer (M11), tests, debugging.
    std::span<const float> tensor(std::string_view name) const;

    // Hot path: filled once in the constructor, from the same storage.
    struct LayerWeights {
        std::span<const float> wq, bq, wk, bk, wv, bv, wo,   // attention
                               w_gate, w_up, w_down,          // MLP
                               ln1, ln2;                      // the two rmsnorm weights
    };
    std::vector<LayerWeights> layers;      // [24]
    std::span<const float> embed_tokens;   // [151936×896] — embedding AND LM head (tied)
    std::span<const float> final_norm;     // model.norm.weight, [896]
};