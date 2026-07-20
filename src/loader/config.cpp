#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "config.h"

using json = nlohmann::json;

static void validate(ModelConfig config) {
    if (config.model_type != "qwen2")
        throw std::runtime_error("Incorrect model type selected, qwen2 expected.");
    if (config.hidden_act != "silu")
        throw std::runtime_error("Incorrect activation function selected, silu expected.");
    if (config.torch_dtype != "bfloat16")
        throw std::runtime_error("Incorrect dtype (precision) selected, bfloat16 expected.");
    if (config.hidden_size % config.num_attention_heads != 0)
        throw std::runtime_error("Hidden size must be divisible by the number of attention heads.");
    if (config.num_attention_heads % config.num_key_value_heads != 0)
        throw std::runtime_error("Number of attention heads must be divisible by the number of key value heads.");
    if (config.use_sliding_window != false)
        throw std::runtime_error("Use sliding window must be false.");
    if (config.tie_word_embeddings != true)
        throw std::runtime_error("Tie word embeddings must be true.");
}

ModelConfig load_config(const std::string& relative_path) {
    std::ifstream config_file(relative_path);

    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file.");
    }

    json config_json;
    try {
        config_file >> config_json;
    }catch (const json::parse_error& e) {
        throw std::runtime_error("Malformed JSON syntax: " + std::string(e.what()));
    }

    try {
        ModelConfig config = config_json.get<ModelConfig>();
        validate(config);

        return config;
    } catch (const json::out_of_range& e) {
        throw std::runtime_error("Config validation failed, a required field is missing: " + std::string(e.what()));
    } catch (const json::type_error& e) {
        throw std::runtime_error("Config type mismatch: " + std::string(e.what()));
    }
}