#include <algorithm>
#include <cmath>
#include <vector>

#include "model.h"
#include "loader/weights.h"
#include "loader/config.h"
#include "check.h"
#include "kernels/kernels.h"

Model::Model(Weights&& w, const ModelConfig& config, size_t max_seq)
    : weights_(std::move(w)),
    config_(config),
    max_seq_(max_seq),
    x_(max_seq_ * config_.hidden_size),
    xb_(max_seq_ * config_.hidden_size),
    xb2_(max_seq_ * config_.hidden_size),
    q_(max_seq_ * config_.hidden_size),
    k_(max_seq_ * static_cast<size_t>(config_.num_key_value_heads) * config_.get_head_dim()),
    v_(max_seq_ * static_cast<size_t>(config_.num_key_value_heads) * config_.get_head_dim()),
    att_(max_seq_ * config_.hidden_size),
    hb_(max_seq_ * config_.intermediate_size),
    hb2_(max_seq_ * config_.intermediate_size),
    logits_(max_seq_ * config_.vocab_size)
{
    INFERNO_CHECK(config_.hidden_size > 0, "Model: hidden_size must be > 0");
    INFERNO_CHECK(config_.vocab_size > 0, "Model: vocab_size must be > 0");
    INFERNO_CHECK(config_.intermediate_size > 0, "Model: intermediate_size must be > 0");
    INFERNO_CHECK(config_.num_attention_heads > 0, "Model: num_attention_heads must be > 0");
    INFERNO_CHECK(config_.num_key_value_heads > 0, "Model: num_key_value_heads must be > 0");
    INFERNO_CHECK(config_.hidden_size % config_.num_attention_heads == 0, "Model: hidden_size must be divisible by num_attention_heads");
    INFERNO_CHECK(config_.num_attention_heads % config_.num_key_value_heads == 0, "Model: num_attention_heads must be divisible by num_key_value_heads");
}

std::span<const float> Model::forward(std::span<const int> tokens) {
    const size_t S { tokens.size() };
    INFERNO_CHECK(S > 0, "Model::forward: tokens must not be empty");
    INFERNO_CHECK(S <= max_seq_, "Model::forward: max sequence size exceeded");

    // Q = H x S = head_dim x n_heads x S
    // K, V = kv_dim x S = head_dim x n_kv_heads x S
    const size_t H { static_cast<size_t>(config_.hidden_size) };
    const size_t I { static_cast<size_t>(config_.intermediate_size) };
    const size_t head_dim { static_cast<size_t>(config_.get_head_dim()) };
    const size_t n_heads { static_cast<size_t>(config_.num_attention_heads) };
    const size_t n_kv_heads { static_cast<size_t>(config_.num_key_value_heads) };
    const size_t kv { n_kv_heads * head_dim };
    const size_t gqa_group_heads { static_cast<size_t>(config_.get_gqa_group_heads()) };
    const float eps { static_cast<float>(config_.rms_norm_eps) };
    const float scale { 1.0f / std::sqrt(static_cast<float>(head_dim)) };

    last_seq_len_ = S; // for get_hidden_state()

    // x[i] = embed_tokens[tokens[i]]
    for (size_t i = 0; i < S; ++i) {
        INFERNO_CHECK(tokens[i] >= 0 && static_cast<size_t>(tokens[i]) < static_cast<size_t>(config_.vocab_size), 
                        "Model::forward: token id {} out of range", tokens[i]);

        std::span<const float> embedding { weights_.embed_tokens.subspan(static_cast<size_t>(tokens[i]) * H, H) };
        std::copy(embedding.begin(), embedding.end(), x_.begin() + i * H);
    }

    // scores over every position attended
    std::vector<float> scores(max_seq_);

    // for each layer
    for (size_t l = 0; l < static_cast<size_t>(config_.num_hidden_layers); ++l) {
        const Weights::LayerWeights& layer { weights_.layers[l] };
        
        // for each token
        for (size_t i = 0; i < S; ++i) {
            std::span<float> xbi { xb_.data() + i * H, H };
            rmsnorm(xbi, { x_.data() + i * H, H }, layer.ln1, eps);
            
            // q[i] = xbi * qw + qb
            linear({ q_.data() + i * H, H }, xbi, layer.qw, layer.qb.data());
            linear({ k_.data() + i * kv, kv }, xbi, layer.kw, layer.kb.data());
            linear({ v_.data() + i * kv, kv }, xbi, layer.vw, layer.vb.data());

            // position
            for (size_t h = 0; h < n_heads; ++h) {
                rope({ q_.data() + i * H + h * head_dim, head_dim }, i, config_.rope_theta);
            }
            for (size_t h = 0; h < n_kv_heads; ++h) {
                rope({ k_.data() + i * kv + h * head_dim, head_dim }, i, config_.rope_theta);
            }

            // GQA attention
            for (size_t h = 0; h < n_heads; ++h) {
                // kv head only increments every gqa_group_heads
                const size_t kv_h { h / gqa_group_heads };
                std::span<const float> q_head { q_.data() + i * H + h * head_dim, head_dim };
                
                // use context of previous keys
                for (size_t j = 0; j <= i; ++j) { 
                    std::span<const float> k_head { k_.data() + j * kv + kv_h * head_dim, head_dim };
                    float dot { 0.0f };
                    for (size_t d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
                    scores[j] = dot * scale;
                }

                // score for each prev token where score_j = (q_i * k_j) x scale
                softmax({ scores.data(), i + 1 });

                std::span<float> out_head { att_.data() + i * H + h * head_dim, head_dim };
                std::fill(out_head.begin(), out_head.end(), 0.0f);

                // aggregate the scores from prev seq into one head and dot v
                for (size_t j = 0; j <= i; ++j) {
                    std::span<const float> v_head { v_.data() + j * kv + kv_h * head_dim, head_dim };
                    const float wj { scores[j] };
                    for (size_t d = 0; d < head_dim; ++d) out_head[d] += wj * v_head[d];
                }
            }

            // SwiGLU
            std::span<float> xb2i { xb2_.data() + i * H, H };
            linear(xb2i, { att_.data() + i * H, H }, layer.ow, nullptr);
            for (size_t d = 0; d < H; ++d) x_[i * H + d] += xb2i[d];

            rmsnorm(xbi, { x_.data() + i * H, H }, layer.ln2, eps);

            std::span<float> gate { hb_.data() + i * I, I };
            std::span<float> up { hb2_.data() + i * I, I };
            linear(up, xbi, layer.up_w, nullptr);
            linear(gate, xbi, layer.gate_w, nullptr);
            silu_mul(gate, up);
            linear(xb2i, gate, layer.down_w, nullptr);
            for (size_t d = 0; d < H; ++d) x_[i * H + d] += xb2i[d];
        }
    }

    std::span<float> xb_last { xb_.data(), H };
    rmsnorm(xb_last, { x_.data() + (S - 1) * H, H }, weights_.final_norm, eps);

    // final dot embed to get similarity with each
    // trying to find token with highest similarity with prediction, higher logit = better
    std::span<float> out_logits { logits_.data(), static_cast<size_t>(config_.vocab_size) };
    linear(out_logits, xb_last, weights_.embed_tokens, nullptr);

    return out_logits;
}

std::span<const float> Model::get_hidden_state() const {
    return { x_.data(), last_seq_len_ * static_cast<size_t>(config_.hidden_size) };
}
