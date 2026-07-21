#pragma once

#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "safetensors.h"
#include "config.h"

class Weights {
    std::unordered_map<std::string, std::vector<float>> data_;
public:
    // Takes ownership of the parsed file; validates every tensor against the
    // config-derived shape table BEFORE converting, then converts BF16 -> fp32.
    Weights(SafeTensorModel&& safetensors_in, const ModelConfig& config);
    Weights(const Weights&) = delete;

    std::span<const float> get_tensor(std::string_view name) const;

    // Layer Weights gotten from _data
    struct LayerWeights {
        std::span<const float> qw, qb, kw, kb, vw, vb, ow, // attention
                               gate_w, up_w, down_w, // mlp
                               ln1, ln2;  // rmsnorm weights
    };

    std::vector<LayerWeights> layers;      // [24]
    std::span<const float> embed_tokens;   // [151936×896] embedding and head (tied)
    std::span<const float> final_norm;     // model.norm.weight [896]
};
