#pragma once

#include "loader/weights.h"
#include "loader/config.h"

class Model {
private:
    Weights weights_;
    ModelConfig config_;
    size_t max_seq_;
    size_t last_seq_len_ { 0 };
    std::vector<float> x_, xb_, xb2_, q_, k_, v_, att_, hb_, hb2_, logits_;
public:
    Model(Weights&& w, const ModelConfig& cfg, size_t max_seq = 2048);

    // Returns logits for last position
    std::span<const float> forward(std::span<const int> tokens);

    std::span<const float> get_hidden_state() const; // X, (seq, H)
};