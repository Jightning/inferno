#pragma once
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

struct ModelConfig
{
    std::vector<std::string> architectures {}; // What model to load
    std::string model_type {}; // Model family
    std::string transformers_version {}; // Version of the Hugging Face transformers library
    
    int hidden_size {}; // Embedding vector size
    int intermediate_size {}; // Peak width of feed forward network
    int num_hidden_layers {}; // Num hidden layers
    int vocab_size {}; // Number of vocab words in tokenizer
    
    int num_attention_heads {}; // Number of attention heads
    int num_key_value_heads {}; // Number of KV heads for GQA
    float attention_dropout {}; // Probability of randomly dropping attention weights during training
    
    int max_position_embeddings {}; // Max context window
    float rope_theta {}; // Theta for RoPE
    bool use_sliding_window {}; // Whether to only look at a certain sliding window of context
    int sliding_window {}; // Window of context token size
    int max_window_layers {}; // Number of attention layers for sliding window

    std::string hidden_act {}; // Activation function
    double rms_norm_eps {}; // To prevent division by 0 during rms
    float initializer_range {};  // Standard deviation of model's weights before training
    bool tie_word_embeddings {}; // Whether to share weights of input embedding with output projection (to and from converting tokens to vectors)

    std::string torch_dtype {}; // Precision
    bool use_cache {}; // Whether to enable KV cache

    int bos_token_id {}; // Beginning of sequence token id
    int eos_token_id {}; // End of sequence token id

    int get_head_dim() const { return hidden_size / num_attention_heads; }; // ndims for each attention head 
    int get_gqa_groups() const { return num_attention_heads / num_key_value_heads; }; // Number of GQA groups
};

// for validation
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ModelConfig, 
    architectures,
    attention_dropout,
    bos_token_id,
    eos_token_id,
    hidden_act,
    hidden_size,
    initializer_range,
    intermediate_size,
    max_position_embeddings,
    max_window_layers,
    model_type,
    num_attention_heads,
    num_hidden_layers,
    num_key_value_heads,
    rms_norm_eps,
    rope_theta,
    sliding_window,
    tie_word_embeddings,
    torch_dtype,
    transformers_version,
    use_cache,
    use_sliding_window,
    vocab_size
)

ModelConfig load_config(const std::string& relative_path);