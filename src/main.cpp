#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "kernels/kernels.h"
#include "loader/config.h"
#include "loader/npy.h"
#include "loader/safetensors.h"
#include "loader/weights.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"

namespace {

void print_usage() {
    std::fprintf(stderr,
        "usage: inferno <command> [options]\n"
        "\n"
        "commands:\n"
        "  generate   generate tokens from a prompt\n"
        "             --model <dir> [--prompt <text> | --token-ids \"1,2,3\"]\n"
        "             [--n-tokens 64] [--max-seq 2048]\n"
        "  parity     check logits and greedy token ids against the HF reference dump\n"
        "             --model <dir> [--data parity_data] [--prompts N] [--steps N]\n"
        "             [--tolerance F] [--max-seq 2048]\n");
}

[[noreturn]] void usage_error(const std::string& message) {
    std::fprintf(stderr, "inferno: %s\n\n", message.c_str());
    print_usage();
    std::exit(2);
}

struct LoadedModel {
    ModelConfig config;
    Model model;
};

LoadedModel load_model(const std::string& model_dir, size_t max_seq) {
    ModelConfig config = load_config(model_dir + "/config.json");
    SafeTensorModel safetensors = load_safetensors(model_dir + "/model.safetensors");

    Weights weights(std::move(safetensors), config);
    Model model(std::move(weights), config, max_seq);

    return LoadedModel{ config, std::move(model) };
}

// Arguments for "generate"
struct GenerateOptions {
    std::string model_dir;
    std::optional<std::string> prompt;
    std::optional<std::vector<int>> token_ids;
    size_t n_tokens = 64;
    size_t max_seq = 2048;
};

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

// Provides the amount of s that is "good" (no incomplete characters)
// incomplete is defined by a lead byte showing a byte length that doesn't match the current length
size_t complete_utf8_prefix_len(std::string_view s) {
    if (s.empty()) return 0;

    // Go backwards from each continuation byte (10...) until a lead byte is found
    size_t back = 0;
    while (back < 3 && back < s.size() && (static_cast<unsigned char>(s[s.size() - 1 - back]) & 0xC0) == 0x80) {
        ++back;
    }

    if (back == s.size()) return 0;

    const size_t lead_pos = s.size() - 1 - back;
    const int len = utf8_seq_len(static_cast<unsigned char>(s[lead_pos]));
    if (len <= 1) return s.size();
    if (lead_pos + static_cast<size_t>(len) <= s.size()) return s.size(); // sequence fully present
    return lead_pos; // incomplete
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

// Arguments for "parity"
struct ParityOptions {
    std::string model_dir;
    std::string data_dir = "parity_data";
    std::optional<size_t> prompts;  // prompts in the manifest
    std::optional<size_t> steps;  // manifest's n_tokens
    std::optional<float> tolerance;  // manifest's logit_tolerance
    size_t max_seq = 2048;
};

ParityOptions parse_parity_args(int argc, char** argv) {
    ParityOptions opts;
    bool have_model = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (i + 1 >= argc) usage_error("missing value for " + std::string(arg));
            return argv[++i];
        };

        if (arg == "--model") { opts.model_dir = next(); have_model = true; }
        else if (arg == "--data") { opts.data_dir = next(); }
        else if (arg == "--prompts") { opts.prompts = parse_size(arg, next()); }
        else if (arg == "--steps") { opts.steps = parse_size(arg, next()); }
        else if (arg == "--max-seq") { opts.max_seq = parse_size(arg, next()); }
        else if (arg == "--tolerance") {
            const std::string_view value = next();
            float tol = 0.0f;
            const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), tol);
            if (ec != std::errc{} || ptr != value.data() + value.size())
                usage_error("--tolerance expects a number, got '" + std::string(value) + "'");
            opts.tolerance = tol;
        }
        else usage_error("unknown flag '" + std::string(arg) + "'");
    }

    if (!have_model) usage_error("--model is required");
    return opts;
}

// max |a[i] - b[i]|
float max_abs_diff(std::span<const float> a, std::span<const float> b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

// Teacher forcing so each step can be compared (wrong at one step doesn't result the rest)
bool run_parity(const ParityOptions& opts) {
    const std::filesystem::path data_dir = opts.data_dir;

    nlohmann::json manifest; // parity manifest generated by the python script
    {
        const std::filesystem::path manifest_path = data_dir / "manifest.json";
        std::ifstream file(manifest_path);
        if (!file.is_open()) throw std::runtime_error("couldn't open " + manifest_path.string());
        file >> manifest;
    }

    const float tolerance = opts.tolerance.value_or(manifest.at("logit_tolerance").get<float>());
    const size_t total_prompts = manifest.at("prompts").size();
    const size_t n_prompts = std::min(opts.prompts.value_or(total_prompts), total_prompts);
    const size_t manifest_steps = manifest.at("n_tokens").get<size_t>();

    std::fprintf(stderr, "parity: loading %s...\n", opts.model_dir.c_str()); // stderr to hide from logging ig
    
    LoadedModel lm = load_model(opts.model_dir, opts.max_seq);
    const size_t vocab_size = static_cast<size_t>(lm.config.vocab_size);

    std::printf("parity: %zu prompt(s), tolerance %.1e, model %s, data %s\n", n_prompts, static_cast<double>(tolerance), opts.model_dir.c_str(), opts.data_dir.c_str());
    std::fflush(stdout);

    size_t prompts_ok = 0;
    float worst_diff_overall = 0.0f;

    // loop through each prompt in the parity
    for (size_t p = 0; p < n_prompts; ++p) {
        const std::string tag = std::format("prompt{:02d}", p);
        const NpyArray tokens_arr = load_npy((data_dir / (tag + "_tokens.npy")).string());
        const NpyArray logits_arr = load_npy((data_dir / (tag + "_logits.npy")).string());

        if (tokens_arr.dtype != NpyDType::I64)
            throw std::runtime_error(tag + "_tokens.npy: expected int64 token ids");
        if (logits_arr.dtype != NpyDType::F32 || logits_arr.shape.size() != 2 || logits_arr.shape[1] != vocab_size)
            throw std::runtime_error(tag + "_logits.npy: expected [steps, " + std::to_string(vocab_size) + "] float32");

        const size_t n_rows = logits_arr.shape[0];
        const size_t prompt_len = tokens_arr.i64.size() - n_rows;
        const size_t steps = std::min(opts.steps.value_or(manifest_steps), n_rows);

        std::vector<int> ids(prompt_len);
        for (size_t i = 0; i < prompt_len; ++i) ids[i] = static_cast<int>(tokens_arr.i64[i]);

        std::fprintf(stderr, "parity: Prompt %02zu, %zu steps, %zu prompt tokens\n", p, steps, prompt_len);

        float worst_diff = 0.0f;
        size_t matched_ids = 0;
        long first_bad_step = -1;
        const auto prompt_start = std::chrono::steady_clock::now();

        // generation loop for the current prompt
        for (size_t step = 0; step < steps; ++step) {
            const std::span<const float> logits = lm.model.forward(ids);
            const std::span<const float> ref_row(logits_arr.f32.data() + step * vocab_size, vocab_size);

            const float diff = max_abs_diff(logits, ref_row);
            const size_t mine = argmax(logits);
            const size_t theirs = static_cast<size_t>(tokens_arr.i64[prompt_len + step]);
            const bool id_ok = (mine == theirs);

            worst_diff = std::max(worst_diff, diff);
            if (id_ok) ++matched_ids;
            if (first_bad_step < 0 && (diff >= tolerance || !id_ok)) first_bad_step = static_cast<long>(step);

            ids.push_back(static_cast<int>(theirs)); // teacher force

            // gonna take a while rn
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - prompt_start).count();
            std::fprintf(stderr, "\r  step %zu/%zu, seq len %zu, %.0fs elapsed", step + 1, steps, ids.size(), elapsed);
            std::fflush(stderr);
        }
        std::fprintf(stderr, "\n");

        const bool prompt_ok = (worst_diff < tolerance) && (matched_ids == steps);
        if (prompt_ok) ++prompts_ok;
        worst_diff_overall = std::max(worst_diff_overall, worst_diff);

        if (prompt_ok) {
            std::printf("prompt: %02zu | worst diff: %.1e | ids: %zu/%zu | OK\n", p, static_cast<double>(worst_diff), matched_ids, steps);
        } else {
            std::printf("prompt: %02zu | worst diff: %.1e | ids: %zu/%zu | FAIL, first at step %ld\n", p, static_cast<double>(worst_diff), matched_ids, steps, first_bad_step);
        }

        std::fflush(stdout);
    }

    const bool all_ok = (prompts_ok == n_prompts);
    std::printf("\n%zu/%zu prompts OK | worst overall diff: %.1e \n", prompts_ok, n_prompts, static_cast<double>(worst_diff_overall));

    return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    try {
        const std::string_view cmd = argv[1];
        if (cmd == "generate") { // predict next n tokens
            run_generate(parse_generate_args(argc, argv));
        } else if (cmd == "parity") { // comparison with Hugging Face (`/scripts/parity.py`)
            if (!run_parity(parse_parity_args(argc, argv))) return 1;
        } else if (cmd == "bench" || cmd == "quantize") {
            // TODO gotta work on these
            std::fprintf(stderr, "inferno: '%s' is not yet implemented\n", argv[1]);
            return 1;
        } else {
            std::fprintf(stderr, "inferno: unknown command '%s'\n\n", argv[1]);
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inferno: %s\n", e.what());
        return 1;
    }

    return 0;
}
