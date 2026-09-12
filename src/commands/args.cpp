#include <charconv>
#include <cstdio>
#include <cstdlib>

#include "args.h"
#include "loader/safetensors.h"
#include "loader/weights.h"

namespace cli {

void print_usage() {
    std::fprintf(stderr,
        "usage: inferno <command> [options]\n"
        "\n"
        "commands:\n"
        "  generate   generate tokens from a prompt\n"
        "             --model <dir> [--prompt <text> | --token-ids \"1,2,3\"]\n"
        "             [--n-tokens 64] [--max-seq 2048]\n"
        "  parity     check logits and greedy token ids against the HF reference\n"
        "             --model <dir> [--data parity_data] [--prompts N] [--steps N]\n"
        "             [--tolerance F] [--max-seq 2048]\n"
        "  bench      measure prefill/decode tok/s and peak RSS, append a row to a CSV\n"
        "             (Release build only; the protocol itself is fixed in src/commands/)\n"
        "             --model <dir> [--config NAME] [--out benchmarks/results.csv]\n"
        "             [--notes TEXT] [--threads 1]\n");
}

[[noreturn]] void usage_error(const std::string& message) {
    std::fprintf(stderr, "inferno: %s\n\n", message.c_str());
    print_usage();
    std::exit(2);
}

LoadedModel load_model(const std::string& model_dir, size_t max_seq) {
    ModelConfig config = load_config(model_dir + "/config.json");
    SafeTensorModel safetensors = load_safetensors(model_dir + "/model.safetensors");

    Weights weights(std::move(safetensors), config);
    Model model(std::move(weights), config, max_seq);

    return LoadedModel{ config, std::move(model) };
}

size_t parse_size(std::string_view flag, std::string_view value) {
    size_t n = 0;
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), n);
    
    if (ec != std::errc{} || ptr != value.data() + value.size()) {
        usage_error(std::string(flag) + " expects a non-negative integer, got '" + std::string(value) + "'");
    }
    
    return n;
}

std::vector<int> parse_token_ids(std::string_view csv) {
    std::vector<int> ids;
    size_t start = 0;

    while (start <= csv.size()) {
        const size_t comma = csv.find(',', start);
        const std::string_view piece = csv.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        int id = 0;
        
        const auto [ptr, ec] = std::from_chars(piece.data(), piece.data() + piece.size(), id);
        if (ec != std::errc{} || ptr != piece.data() + piece.size()) {
            usage_error("--token-ids expects a comma-separated list of integers, got '" + std::string(piece) + "'");
        }
        
        ids.push_back(id);
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }

    return ids;
}

}  // namespace cli
