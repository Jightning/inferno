#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "loader/config.h"

using json = nlohmann::json;

namespace {

// mock json
json base_config() {
    return json{
        {"architectures", json::array({"Qwen2ForCausalLM"})},
        {"model_type", "qwen2"},
        {"transformers_version", "4.43.1"},

        {"hidden_size", 896},
        {"intermediate_size", 4864},
        {"num_hidden_layers", 24},
        {"vocab_size", 151936},

        {"num_attention_heads", 14},
        {"num_key_value_heads", 2},
        {"attention_dropout", 0.0},

        {"max_position_embeddings", 32768},
        {"rope_theta", 1000000.0},
        {"use_sliding_window", false},
        {"sliding_window", 32768},
        {"max_window_layers", 21},

        {"hidden_act", "silu"},
        {"rms_norm_eps", 1e-06},
        {"initializer_range", 0.02},
        {"tie_word_embeddings", true},

        {"torch_dtype", "bfloat16"},
        {"use_cache", true},

        {"bos_token_id", 151643},
        {"eos_token_id", 151645},
    };
}

class TempJson {
public:
    explicit TempJson(const std::string& text) {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() / ("inferno_test_config_" + std::to_string(counter++) + ".json");
        std::ofstream out(path_);
        out << text;
    }


    ~TempJson() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    TempJson(const TempJson&) = delete;
    TempJson& operator=(const TempJson&) = delete;

    std::string str() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

}

TEST_CASE("load_config accepts a valid config") {
    TempJson config_file(base_config().dump());
    ModelConfig config = load_config(config_file.str());

    REQUIRE(config.architectures.size() == 1);
    CHECK(config.architectures[0] == "Qwen2ForCausalLM");
    CHECK(config.model_type == "qwen2");
    CHECK(config.transformers_version == "4.43.1");

    CHECK(config.hidden_size == 896);
    CHECK(config.intermediate_size == 4864);
    CHECK(config.num_hidden_layers == 24);
    CHECK(config.vocab_size == 151936);

    CHECK(config.num_attention_heads == 14);
    CHECK(config.num_key_value_heads == 2);
    CHECK(config.attention_dropout == doctest::Approx(0.0));

    CHECK(config.max_position_embeddings == 32768);
    CHECK(config.rope_theta == doctest::Approx(1000000.0));
    CHECK(config.use_sliding_window == false);
    CHECK(config.sliding_window == 32768);
    CHECK(config.max_window_layers == 21);

    CHECK(config.hidden_act == "silu");
    CHECK(config.rms_norm_eps == doctest::Approx(1e-06));
    CHECK(config.initializer_range == doctest::Approx(0.02));
    CHECK(config.tie_word_embeddings == true);

    CHECK(config.torch_dtype == "bfloat16");
    CHECK(config.use_cache == true);

    CHECK(config.bos_token_id == 151643);
    CHECK(config.eos_token_id == 151645);

    CHECK(config.get_head_dim() == 64);
    CHECK(config.get_gqa_groups() == 7);
}

TEST_CASE("load_config rejects a missing required field") {
    json config_json = base_config();
    config_json.erase("hidden_size");
    TempJson config_file(config_json.dump());

    CHECK_THROWS_AS(load_config(config_file.str()), std::runtime_error);
}

TEST_CASE("load_config rejects a field of the wrong type") {
    json config_json = base_config();
    config_json["hidden_size"] = "big";
    TempJson config_file(config_json.dump());

    CHECK_THROWS_AS(load_config(config_file.str()), std::runtime_error);
}

TEST_CASE("load_config rejects malformed JSON") {
    TempJson config_file("{\"hidden_size\": 896");

    CHECK_THROWS_AS(load_config(config_file.str()), std::runtime_error);
}

TEST_CASE("load_config enforces the validation rules") {
    json config_json = base_config();

    SUBCASE("wrong model type") {
        config_json["model_type"] = "llama";
    }

    SUBCASE("hidden size not divisible by attention heads") {
        config_json["hidden_size"] = 897;
    }

    SUBCASE("attention heads not divisible by kv heads") {
        config_json["num_key_value_heads"] = 3;
    }

    TempJson config_file(config_json.dump());
    CHECK_THROWS_AS(load_config(config_file.str()), std::runtime_error);
}

TEST_CASE("load_config throws when the file is missing") {
    std::filesystem::path missing = std::filesystem::temp_directory_path() / "inferno_no_such_config.json";
    CHECK_THROWS_WITH_AS(load_config(missing.string()), "Could not open config file.", std::runtime_error);
}
