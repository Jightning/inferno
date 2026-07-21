#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>

#include "check.h"
#include "weights.h"

namespace {

constexpr size_t kBf16Bytes = 2;

std::string shape_to_string(const std::vector<size_t>& shape) {
    std::string out = "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(shape[i]);
    }

    return out + "]";
}

// All the expected shapes for the safetensor along with the expected names
std::unordered_map<std::string, std::vector<size_t>> expected_shapes(const ModelConfig& config) {
    const size_t hidden = config.hidden_size;
    const size_t inter = config.intermediate_size;
    const size_t vocab = config.vocab_size;
    const size_t kv_dim = static_cast<size_t>(config.num_key_value_heads) * config.get_head_dim();

    std::unordered_map<std::string, std::vector<size_t>> expected;
    expected["model.embed_tokens.weight"] = {vocab, hidden};
    expected["model.norm.weight"] = {hidden};

    for (int i = 0; i < config.num_hidden_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";

        expected[prefix + "input_layernorm.weight"] = {hidden};
        expected[prefix + "post_attention_layernorm.weight"] = {hidden};

        expected[prefix + "self_attn.q_proj.weight"] = {hidden, hidden};
        expected[prefix + "self_attn.q_proj.bias"] = {hidden};
        expected[prefix + "self_attn.k_proj.weight"] = {kv_dim, hidden};
        expected[prefix + "self_attn.k_proj.bias"] = {kv_dim};
        expected[prefix + "self_attn.v_proj.weight"] = {kv_dim, hidden};
        expected[prefix + "self_attn.v_proj.bias"] = {kv_dim};
        expected[prefix + "self_attn.o_proj.weight"] = {hidden, hidden};

        expected[prefix + "mlp.gate_proj.weight"] = {inter, hidden};
        expected[prefix + "mlp.up_proj.weight"] = {inter, hidden};
        expected[prefix + "mlp.down_proj.weight"] = {hidden, inter};
    }

    return expected;
}

float bf16_to_f32(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

// reinterpret_cast here would be UB, byte is not guaranteed to land on a nice boundary
std::uint16_t load_bf16(const std::byte* p) {
    std::uint16_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

// list of raw BF16 bytes to FP32
std::vector<float> raw_to_f32(std::span<const std::byte> raw) {
    std::vector<float> out(raw.size() / kBf16Bytes);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = bf16_to_f32(load_bf16(raw.data() + i * kBf16Bytes));
        
    return out;
}

}  // namespace

Weights::Weights(SafeTensorModel&& safetensors_in, const ModelConfig& config) {
    SafeTensorModel safetensors = std::move(safetensors_in);

    // Error handling
    const auto expected = expected_shapes(config);
    const auto file = safetensors.mmap_file.get_bytes();

    INFERNO_CHECK(safetensors.tensors.size() == expected.size(),
                "expected {} tensors, the safetensors file has {}", expected.size(), safetensors.tensors.size());

    for (const auto& [tensor_name, expected_shape] : expected) {
        // names are already set, so it should be expected to exist
        const auto found = safetensors.tensors.find(tensor_name);
        INFERNO_CHECK(found != safetensors.tensors.end(),
                    "tensor '{}': missing from the safetensors file", tensor_name);
        
        const SafeTensor& safetensor = found->second;

        // checking values line up to what we're going to be processing
        INFERNO_CHECK(safetensor.dtype == "BF16",
                    "tensor '{}': expected dtype BF16, file has {}", tensor_name, safetensor.dtype);
        INFERNO_CHECK(safetensor.shape == expected_shape,
                    "tensor '{}': expected shape {}, file has {}", tensor_name, shape_to_string(expected_shape),  shape_to_string(safetensor.shape));

        // making sure the header's byte range is good
        const size_t elements = std::accumulate(expected_shape.begin(), expected_shape.end(), size_t{1}, std::multiplies<>{});
        INFERNO_CHECK(safetensor.end >= safetensor.start && (safetensor.end - safetensor.start) == elements * kBf16Bytes,
                    "tensor '{}': shape {} needs {} bytes, header declares {}", tensor_name, shape_to_string(expected_shape), elements * kBf16Bytes, safetensor.end - safetensor.start);
        INFERNO_CHECK(safetensors.data_start <= file.size() && safetensor.end <= file.size() - safetensors.data_start,
                    "tensor '{}': data range ends past the end of the file", tensor_name);
    }

    // Converting bf16 to fp32 and passing to data_
    data_.reserve(expected.size());
    //                              safetensor, naming it such is annoying to look at
    for (const auto& [tensor_name, stensor] : safetensors.tensors) {
        const size_t nbytes = stensor.end - stensor.start;

        std::span<const std::byte> raw = file.subspan(safetensors.data_start + stensor.start, nbytes);
        data_[tensor_name] = raw_to_f32(raw);

    }

    // defining all the layers with the needed information
    layers.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        LayerWeights& layer = layers[i];

        layer.qw = get_tensor(prefix + "self_attn.q_proj.weight"); // query weights
        layer.qb = get_tensor(prefix + "self_attn.q_proj.bias"); // query bias
        layer.kw = get_tensor(prefix + "self_attn.k_proj.weight"); // key weights
        layer.kb = get_tensor(prefix + "self_attn.k_proj.bias"); // key bias
        layer.vw = get_tensor(prefix + "self_attn.v_proj.weight"); // value weights
        layer.vb = get_tensor(prefix + "self_attn.v_proj.bias"); // value bias
        layer.ow = get_tensor(prefix + "self_attn.o_proj.weight"); // output weights

        layer.gate_w = get_tensor(prefix + "mlp.gate_proj.weight");
        layer.up_w = get_tensor(prefix + "mlp.up_proj.weight");
        layer.down_w = get_tensor(prefix + "mlp.down_proj.weight");

        layer.ln1 = get_tensor(prefix + "input_layernorm.weight");
        layer.ln2 = get_tensor(prefix + "post_attention_layernorm.weight");
    }

    embed_tokens = get_tensor("model.embed_tokens.weight");
    final_norm = get_tensor("model.norm.weight");
}

// get tensor by the given name
std::span<const float> Weights::get_tensor(std::string_view name) const {
    const auto found = data_.find(std::string(name));
    INFERNO_CHECK(found != data_.end(), "tensor '{}': not loaded", name); // tens
    return found->second;
}
