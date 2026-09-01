#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "loader/config.h"
#include "loader/npy.h"
#include "loader/safetensors.h"
#include "loader/weights.h"
#include "model/model.h"

namespace {

const std::filesystem::path kModelDir = INFERNO_MODEL_DIR;
const std::filesystem::path kConfigPath = kModelDir / "config.json";
const std::filesystem::path kWeightsPath = kModelDir / "model.safetensors";
const std::filesystem::path kParityDir = PARITY_DATA_DIR;

bool model_present() {
    return std::filesystem::exists(kConfigPath) && std::filesystem::exists(kWeightsPath);
}

// The four checkpoint dumps (embed/block0/final_norm, on top of the base
// tokens/logits pair) come from `parity.py --dump-intermediates` and are not
// guaranteed to exist alongside the rest of parity_data/.
bool checkpoints_present() {
    for (const char* name : {"prompt00_tokens.npy", "prompt00_logits.npy", "prompt00_embed.npy",
                              "prompt00_block0.npy", "prompt00_final_norm.npy"}) {
        if (!std::filesystem::exists(kParityDir / name)) return false;
    }
    return true;
}

Weights load_weights(const ModelConfig& config) {
    return Weights(load_safetensors(kWeightsPath.string()), config);
}

// max |a - b| over the whole span, with the index of the worst element.
std::pair<float, size_t> max_abs_diff(std::span<const float> a, std::span<const float> b) {
    REQUIRE(a.size() == b.size());
    float worst = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::abs(a[i] - b[i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    return {worst, worst_i};
}

// prompt00's prompt-only token ids (everything before the first generated token).
std::vector<int> prompt00_ids() {
    const NpyArray tokens = load_npy((kParityDir / "prompt00_tokens.npy").string());
    const NpyArray logits = load_npy((kParityDir / "prompt00_logits.npy").string());
    const size_t prompt_len = tokens.i64.size() - logits.shape[0];

    std::vector<int> ids(prompt_len);
    for (size_t i = 0; i < prompt_len; ++i) ids[i] = static_cast<int>(tokens.i64[i]);
    return ids;
}

}  // namespace

// Deliberately flat rather than SUBCASEs (see tests/loader/weights.cpp): doctest
// re-enters the test body once per subcase, which would reload and reconvert the
// whole 943 MiB checkpoint each time.
TEST_CASE("Model rejects bad forward() inputs") {
    if (!model_present()) {
        MESSAGE("skipped: no checkpoint at " << kModelDir.string());
        return;
    }

    const ModelConfig config = load_config(kConfigPath.string());
    Model model(load_weights(config), config, /*max_seq=*/4);

    // S > max_seq.
    const std::vector<int> too_long = {0, 0, 0, 0, 0};
    CHECK_THROWS_AS(model.forward(too_long), std::runtime_error);

    // Token id == vocab_size, one past the last valid id.
    const std::vector<int> out_of_range = {config.vocab_size};
    CHECK_THROWS_AS(model.forward(out_of_range), std::runtime_error);

    const std::vector<int> empty;
    CHECK_THROWS_AS(model.forward(empty), std::runtime_error);
}

// M6 checkpoint A. `num_hidden_layers` is overridden to 0 on a *copy* of the real
// config passed to Model -- Weights itself is still built from the real config, so
// it loads all 24 layers correctly, but Model's layer loop (bounded by
// config_.num_hidden_layers) never runs. The tail (final rmsnorm + LM head) writes
// into xb_/logits_, never x_, so hidden_state() afterward is exactly the embedding
// lookup with nothing else applied.
TEST_CASE("Model checkpoint A: embedding matches prompt00_embed.npy") {
    if (!model_present()) {
        MESSAGE("skipped: no checkpoint at " << kModelDir.string());
        return;
    }
    if (!checkpoints_present()) {
        MESSAGE("skipped: no checkpoint fixtures at " << kParityDir.string());
        return;
    }

    const ModelConfig config = load_config(kConfigPath.string());
    const std::vector<int> ids = prompt00_ids();
    const size_t H = static_cast<size_t>(config.hidden_size);

    ModelConfig embed_only = config;
    embed_only.num_hidden_layers = 0;
    Model model(load_weights(config), embed_only, ids.size());

    model.forward(ids);
    const std::span<const float> hidden = model.get_hidden_state();

    const NpyArray expected = load_npy((kParityDir / "prompt00_embed.npy").string());
    REQUIRE(expected.shape == std::vector<size_t>{ids.size(), H});

    const auto [diff, idx] = max_abs_diff(hidden, expected.f32);
    CAPTURE(idx);
    // It's a copy; anything larger means the row indexing is wrong.
    CHECK(diff < 1e-6f);
}

// M6 checkpoint B. Same trick, `num_hidden_layers = 1`, so the layer loop runs
// exactly block 0 and stops. This is the checkpoint that exercises rope, GQA, the
// attention scale, and the QKV biases all at once, with exactly one block to search
// if it fails.
TEST_CASE("Model checkpoint B: block 0 matches prompt00_block0.npy") {
    if (!model_present()) {
        MESSAGE("skipped: no checkpoint at " << kModelDir.string());
        return;
    }
    if (!checkpoints_present()) {
        MESSAGE("skipped: no checkpoint fixtures at " << kParityDir.string());
        return;
    }

    const ModelConfig config = load_config(kConfigPath.string());
    const std::vector<int> ids = prompt00_ids();
    const size_t H = static_cast<size_t>(config.hidden_size);

    ModelConfig one_block = config;
    one_block.num_hidden_layers = 1;
    Model model(load_weights(config), one_block, ids.size());

    model.forward(ids);
    const std::span<const float> hidden = model.get_hidden_state();

    const NpyArray expected = load_npy((kParityDir / "prompt00_block0.npy").string());
    REQUIRE(expected.shape == std::vector<size_t>{ids.size(), H});

    const auto [diff, idx] = max_abs_diff(hidden, expected.f32);
    CAPTURE(idx);
    // Roadmap: "Expect < ~1e-5" -- not a hard target, some fp32-accumulation slack
    // over rmsnorm's 896-wide reduction is expected here. 1e-3 is the "real bug"
    // floor per the magnitudes table; stay well under it.
    CHECK(diff < 3e-5f);
}

// M6 checkpoints C and D together, from one real (unmodified, 24-layer) forward
// pass -- it computes both in a single call, so there's no reason to load twice.
//
// C: Model only applies the final rmsnorm to the last position (the "single
// biggest avoidable cost" optimization the roadmap calls out), so hidden_state()
// after a full forward is the pre-final-norm residual stream for every position.
// To compare against the HF dump (which is post-norm, every position), rmsnorm is
// applied by hand here using the same weight the model used internally -- captured
// from `weights` *before* it's moved into Model. That span stays valid across the
// move: Weights' move constructor is defaulted, so its data_ map is relocated
// (bucket/node ownership transfers) without touching the individual
// std::vector<float> heap buffers the spans point into -- the same guarantee
// Weights' own `layers` spans already depend on, and main.cpp's
// `Model{std::move(weights), ...}` already exercises it successfully.
//
// D: the actual production call -- forward() on the prompt, compared against row 0
// of prompt00_logits.npy (the distribution that produced the first generated
// token), plus an exact greedy-token check.
TEST_CASE("Model checkpoints C & D: final norm and logits match prompt00") {
    if (!model_present()) {
        MESSAGE("skipped: no checkpoint at " << kModelDir.string());
        return;
    }
    if (!checkpoints_present()) {
        MESSAGE("skipped: no checkpoint fixtures at " << kParityDir.string());
        return;
    }

    const ModelConfig config = load_config(kConfigPath.string());
    const std::vector<int> ids = prompt00_ids();
    const size_t H = static_cast<size_t>(config.hidden_size);
    const float eps = static_cast<float>(config.rms_norm_eps);

    Weights weights = load_weights(config);
    const std::span<const float> final_norm_w = weights.final_norm;
    Model model(std::move(weights), config, ids.size());

    const std::span<const float> logits = model.forward(ids);
    const std::span<const float> hidden = model.get_hidden_state();  // post-block-23, pre-final-norm

    // Checkpoint C.
    const NpyArray expected_norm = load_npy((kParityDir / "prompt00_final_norm.npy").string());
    REQUIRE(expected_norm.shape == std::vector<size_t>{ids.size(), H});

    std::vector<float> normed(hidden.size());
    for (size_t s = 0; s < ids.size(); ++s) {
        rmsnorm({normed.data() + s * H, H}, hidden.subspan(s * H, H), final_norm_w, eps);
    }
    const auto [c_diff, c_idx] = max_abs_diff(normed, expected_norm.f32);
    CAPTURE(c_idx);
    // Roadmap: "Expect < ~1e-4" -- divergence grows with depth, and this is fp32
    // accumulation over 24 stacked layers, not a bug (magnitudes table: 1e-5-1e-3
    // is expected here; 1e-3+ would be the real-bug band, same gate D uses below).
    CHECK(c_diff < 1e-3f);

    // Checkpoint D.
    const std::vector<int64_t> all_tokens = load_npy_i64((kParityDir / "prompt00_tokens.npy").string());
    const NpyArray ref_logits = load_npy((kParityDir / "prompt00_logits.npy").string());
    REQUIRE(ref_logits.shape.size() == 2);
    REQUIRE(ref_logits.shape[1] == static_cast<size_t>(config.vocab_size));

    const std::span<const float> row0 =
        std::span<const float>(ref_logits.f32).first(static_cast<size_t>(config.vocab_size));
    const auto [d_diff, d_idx] = max_abs_diff(logits, row0);
    CAPTURE(d_idx);
    CHECK(d_diff < 1e-3f);

    // Matching ids matter more than matching floats.
    REQUIRE(all_tokens.size() > ids.size());
    CHECK(argmax(logits) == static_cast<size_t>(all_tokens[ids.size()]));
}
