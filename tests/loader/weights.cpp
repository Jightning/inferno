#include <doctest/doctest.h>

#include <filesystem>
#include <string>

#include "loader/weights.h"

namespace {

const std::filesystem::path kModelDir = INFERNO_MODEL_DIR;
const std::filesystem::path kConfigPath = kModelDir / "config.json";
const std::filesystem::path kWeightsPath = kModelDir / "model.safetensors";

bool model_present() {
    return std::filesystem::exists(kConfigPath) && std::filesystem::exists(kWeightsPath);
}

// model.embed_tokens.weight[0, :4], read straight out of the checkpoint header by
// python and widened BF16 -> fp32. Exact, not approximate: BF16 is the top half of
// an fp32, so any difference at all means the offset math is wrong.
constexpr float kEmbedRow0[4] = {
    -0.0103759765625f,
    0.040771484375f,
    0.00970458984375f,
    6.961822509765625e-05f,
};

}  // namespace

// Integration smoke: the real 943 MiB checkpoint, end to end. Skipped when the
// model is absent so CI (which has no checkpoint) still passes.
TEST_CASE("Weights loads the real checkpoint") {
    if (!model_present()) {
        MESSAGE("skipped: no checkpoint at " << kModelDir.string());
        return;
    }

    const ModelConfig config = load_config(kConfigPath.string());

    SafeTensorModel safetensors = load_safetensors(kWeightsPath.string());
    CHECK(safetensors.tensors.size() == 290);

    const Weights weights(std::move(safetensors), config);

    // Deliberately flat rather than SUBCASEs: doctest re-enters the test body once
    // per subcase, which would reload and reconvert the whole checkpoint each time.

    // The embedding values are the smoke test proper — they only come out right if
    // the header parse, the data_start base, the per-tensor offsets, and the BF16
    // widening are all correct.
    REQUIRE(weights.embed_tokens.size() ==
            static_cast<size_t>(config.vocab_size) * config.hidden_size);
    for (size_t i = 0; i < 4; ++i) {
        CAPTURE(i);
        CHECK(weights.embed_tokens[i] == kEmbedRow0[i]);
    }

    // The named spans must alias the same storage get_tensor hands out.
    CHECK(weights.final_norm.size() == static_cast<size_t>(config.hidden_size));
    CHECK(weights.final_norm.data() == weights.get_tensor("model.norm.weight").data());
    CHECK(weights.embed_tokens.data() == weights.get_tensor("model.embed_tokens.weight").data());

    REQUIRE(weights.layers.size() == static_cast<size_t>(config.num_hidden_layers));

    const size_t hidden = config.hidden_size;
    const size_t inter = config.intermediate_size;
    const size_t kv_dim = static_cast<size_t>(config.num_key_value_heads) * config.get_head_dim();

    for (size_t i = 0; i < weights.layers.size(); ++i) {
        CAPTURE(i);
        const Weights::LayerWeights& layer = weights.layers[i];

        // k/v are narrower than q — that asymmetry is GQA, and getting it wrong is
        // the shape bug that surfaces as garbage logits rather than a crash.
        CHECK(layer.qw.size() == hidden * hidden);
        CHECK(layer.qb.size() == hidden);
        CHECK(layer.kw.size() == kv_dim * hidden);
        CHECK(layer.kb.size() == kv_dim);
        CHECK(layer.vw.size() == kv_dim * hidden);
        CHECK(layer.vb.size() == kv_dim);
        CHECK(layer.ow.size() == hidden * hidden);

        CHECK(layer.gate_w.size() == inter * hidden);
        CHECK(layer.up_w.size() == inter * hidden);
        CHECK(layer.down_w.size() == hidden * inter);

        CHECK(layer.ln1.size() == hidden);
        CHECK(layer.ln2.size() == hidden);

        // Each layer must be wired to its own tensors, not layer 0's.
        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        CHECK(layer.qw.data() == weights.get_tensor(prefix + "self_attn.q_proj.weight").data());
        CHECK(layer.ln2.data() == weights.get_tensor(prefix + "post_attention_layernorm.weight").data());
    }

    // The mapping is released when the constructor's SafeTensorModel dies, so
    // readable data here proves Weights owns its own fp32 copy.
    CHECK(weights.layers[0].ln1[0] ==
          weights.get_tensor("model.layers.0.input_layernorm.weight")[0]);
}

TEST_CASE("Weights names the offending tensor when a shape disagrees") {
    if (!model_present()) {
        MESSAGE("skipped: no checkpoint at " << kModelDir.string());
        return;
    }

    ModelConfig config = load_config(kConfigPath.string());
    config.intermediate_size += 1;  // now every MLP tensor's expected shape is wrong

    std::string message;
    try {
        Weights weights(load_safetensors(kWeightsPath.string()), config);
        FAIL("expected the shape check to reject the mismatched config");
    } catch (const std::runtime_error& e) {
        message = e.what();
    }

    // The whole point of the check is that it identifies *which* tensor is wrong.
    CHECK(message.find("model.layers.") != std::string::npos);
    CHECK(message.find("_proj.weight") != std::string::npos);
    CHECK(message.find("shape") != std::string::npos);
}

TEST_CASE("Weights rejects a config whose tensor set does not match the file") {
    if (!model_present()) {
        MESSAGE("skipped: no checkpoint at " << kModelDir.string());
        return;
    }

    ModelConfig config = load_config(kConfigPath.string());
    config.num_hidden_layers -= 1;  // 12 fewer tensors than the file holds

    CHECK_THROWS_AS(Weights(load_safetensors(kWeightsPath.string()), config),
                    std::runtime_error);
}
