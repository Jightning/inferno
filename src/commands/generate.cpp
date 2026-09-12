#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "args.h"
#include "generate.h"
#include "kernels/kernels.h"
#include "tokenizer/tokenizer.h"

namespace generate {
namespace {

using cli::LoadedModel;
using cli::load_model;
using cli::parse_size;
using cli::parse_token_ids;
using cli::usage_error;

// Arguments for "generate"
struct GenerateOptions {
    std::string model_dir;
    std::optional<std::string> prompt;
    std::optional<std::vector<int>> token_ids;
    size_t n_tokens = 64;
    size_t max_seq = 2048;
};

GenerateOptions parse_generate_args(int argc, char** argv) {
    GenerateOptions opts;
    bool have_model = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view { // lambda to check each argument and then move up
            if (i + 1 >= argc) usage_error("missing value for " + std::string(arg));
            return argv[++i];
        };

        if (arg == "--model") { opts.model_dir = next(); have_model = true; }
        else if (arg == "--prompt") { opts.prompt = std::string(next()); }
        else if (arg == "--token-ids") { opts.token_ids = parse_token_ids(next()); }
        else if (arg == "--n-tokens") { opts.n_tokens = parse_size(arg, next()); }
        else if (arg == "--max-seq") { opts.max_seq = parse_size(arg, next()); }
        else usage_error("unknown flag '" + std::string(arg) + "'");
    }

    if (!have_model) usage_error("--model is required");
    if (opts.prompt.has_value() == opts.token_ids.has_value()) {
        usage_error("exactly one of --prompt or --token-ids is required");
    }

    return opts;
}

int utf8_seq_len(unsigned char seq) {
    if (seq < 0x80) return 1;
    if ((seq & 0xE0) == 0xC0) return 2;
    if ((seq & 0xF0) == 0xE0) return 3;
    if ((seq & 0xF8) == 0xF0) return 4;

    return -1;
}

void run_generate(const GenerateOptions& opts) {
    const Tokenizer tokenizer(opts.model_dir + "/tokenizer.json");
    LoadedModel model = load_model(opts.model_dir, opts.max_seq);

    std::vector<int> ids = opts.token_ids ? *opts.token_ids : tokenizer.encode(*opts.prompt);

    std::string pending;
    for (size_t step = 0; step < opts.n_tokens && ids.size() < opts.max_seq; ++step) {
        const std::span<const float> logits = model.model.forward(ids);
        const int next = static_cast<int>(argmax(logits));
        if (next == model.config.eos_token_id) break;

        ids.push_back(next);
        pending += tokenizer.decode(std::span(&next, 1));

        // printing the complete parts of the curren pending
        const size_t good = complete_utf8_prefix_len(pending);
        std::cout << pending.substr(0, good) << std::flush;
        pending.erase(0, good);
    }

    std::cout << pending << '\n';
}

}  // namespace

size_t complete_utf8_prefix_len(std::string_view s) {
    if (s.empty()) return 0;

    // Go backwards from each continuation byte (10) until a lead byte is found
    size_t back = 0;
    while (back < 3 && back < s.size() && (static_cast<unsigned char>(s[s.size() - 1 - back]) & 0xC0) == 0x80) {
        ++back;
    }

    if (back == s.size()) return 0;

    const size_t lead_pos = s.size() - 1 - back;
    const int len = utf8_seq_len(static_cast<unsigned char>(s[lead_pos]));
    if (len <= 1) return s.size();
    if (lead_pos + static_cast<size_t>(len) <= s.size()) return s.size(); // sequence fully present
    return lead_pos; // means its incomplete
}

void run(int argc, char** argv) {
    run_generate(parse_generate_args(argc, argv));
}

}  // namespace generate
