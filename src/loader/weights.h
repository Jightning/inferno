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
    // Validates every tensor against config then converts BF16 -> fp32.
    Weights(SafeTensorModel&& safetensors_in, const ModelConfig& config);

    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;
    Weights(Weights&&) = default;
    Weights& operator=(Weights&&) = default;

    std::span<const float> get_tensor(std::string_view name) const;

    // Layer Weights gotten from _data
    struct LayerWeights {
        std::span<const float> qw, qb, kw, kb, vw, vb, ow, // attention
                                gate_w, up_w, down_w, // mlp
                                ln1, ln2;  // rmsnorm weights
    };

    std::vector<LayerWeights> layers;
    std::span<const float> embed_tokens; 
    std::span<const float> final_norm;
};
