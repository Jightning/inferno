#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <string_view>

#include <sys/resource.h>  // getrusage, POSIX (Linux + macOS)

#include "args.h"
#include "bench.h"
#include "check.h"
#include "kernels/kernels.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"

namespace bench {

const char* const kPrompt =
    "The history of computing spans centuries, from mechanical calculators built in "
    "the seventeenth century to the general-purpose electronic machines that emerged "
    "during the Second World War, and eventually to the integrated circuits and "
    "microprocessors that power everything from wristwatches to spacecraft today, "
    "carrying every calculation through a chain of transistors switching billions "
    "of times each second.";

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Trimmed first line of a shell command, or "" if it produced nothing.
std::string first_line_of(const char* command) {
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) return {};

    char buf[128] = {};
    const char* line = std::fgets(buf, sizeof buf, pipe);
    pclose(pipe);
    if (line == nullptr) return {};

    std::string out(line);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

// [Hello "World"] -> ["Hello ""World"""]
std::string csv_quote(std::string_view field) {
    std::string out = "\"";
    for (const char c : field) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

}  // namespace

Timings time_generation(Model& model, std::span<const int> prompt_ids, size_t decode_tokens) {
    std::vector<int> ids(prompt_ids.begin(), prompt_ids.end());

    // Prefill
    const auto prefill_start = std::chrono::steady_clock::now();
    const std::span<const float> prefill_logits = model.forward(ids);
    const double prefill_sec = seconds_since(prefill_start);
    ids.push_back(static_cast<int>(argmax(prefill_logits)));

    // Decode
    const auto decode_start = std::chrono::steady_clock::now();
    for (size_t step = 0; step < decode_tokens; ++step) {
        const std::span<const float> logits = model.forward(ids);
        ids.push_back(static_cast<int>(argmax(logits)));

        std::fprintf(stderr, "\r    decode %zu/%zu, seq len %zu, %.0fs elapsed",
                     step + 1, decode_tokens, ids.size(), seconds_since(decode_start));
        std::fflush(stderr);
    }
    const double decode_sec = seconds_since(decode_start);
    std::fprintf(stderr, "\n");

    return {prefill_sec, decode_sec};
}

void require_release_build() {
#ifndef NDEBUG
    throw std::runtime_error("bench: reconfigure with -DCMAKE_BUILD_TYPE=Release");
#endif
}

Result measure(Model& model, std::span<const int> prompt_ids) {
    require_release_build();

    std::fprintf(stderr, "bench: %zu prompt tokens, %zu decode tokens, %zu runs (median)\n",
                 prompt_ids.size(), kDecodeTokens, kMeasuredRuns);

    // Discarded for the warmup
    std::fprintf(stderr, "  warm-up...\n");
    static_cast<void>(time_generation(model, prompt_ids, kWarmupDecodeTokens));

    std::vector<double> prefill_rates, decode_rates;
    for (size_t run = 0; run < kMeasuredRuns; ++run) {
        std::fprintf(stderr, "  run %zu/%zu...\n", run + 1, kMeasuredRuns);

        const Timings t = time_generation(model, prompt_ids, kDecodeTokens);
        const double prefill_tps = static_cast<double>(prompt_ids.size()) / t.prefill_sec;
        const double decode_tps = static_cast<double>(kDecodeTokens) / t.decode_sec;

        std::fprintf(stderr, "  run %zu/%zu: prefill %.1fs (%.2f tok/s), decode %.0fs (%.3f tok/s)\n",
                    run + 1, kMeasuredRuns, t.prefill_sec, prefill_tps, t.decode_sec, decode_tps);

        prefill_rates.push_back(prefill_tps);
        decode_rates.push_back(decode_tps);
    }

    return { median(prefill_rates), median(decode_rates), peak_rss_mb() };
}

double median(std::vector<double> values) {
    INFERNO_CHECK(!values.empty(), "bench::median: no values");

    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return (n % 2 == 1) ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

// peak-RSS field has different units on Linux and macOS.
double peak_rss_mb() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);  // bytes -> MB
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;  // KiB -> MB
#endif
}

std::string current_date() {
    const auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    return std::format("{:%Y-%m-%d}", today);
}

std::string current_git_hash() {
    const std::string hash = first_line_of("git rev-parse --short HEAD 2>/dev/null");
    if (hash.empty()) return "unknown";

    const bool dirty = !first_line_of("git status --porcelain 2>/dev/null").empty();
    return dirty ? hash + "-dirty" : hash;
}

std::string format_csv_row(const Row& row) {
    INFERNO_CHECK(row.config.find_first_of(",\"\n") == std::string::npos, "bench: --config can't contain a comma, quote, or newline");

    return std::format("{},{},{},{},{:.2f},{:.2f},{:.0f},{}\n",
                        row.git_hash, row.date, row.config, row.threads,
                        row.prefill_tps, row.decode_tps, row.peak_rss_mb, csv_quote(row.notes));
}

void append_csv_row(const std::string& path_str, const Row& row) {
    const std::string line = format_csv_row(row);  // validate before touching the file

    const std::filesystem::path path = path_str;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());

    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    const bool need_header = ec || size == 0;

    std::ofstream out(path, std::ios::app);
    INFERNO_CHECK(out.is_open(), "bench: could not open {} for writing", path.string());

    if (need_header) out << kCsvHeader << '\n';
    out << line;
}

namespace {

using cli::LoadedModel;
using cli::load_model;
using cli::parse_size;
using cli::usage_error;

// Arguments for "bench"
struct BenchOptions {
    std::string model_dir;
    std::string config_name = "fp32-nocache";
    std::string out_path = "benchmarks/results.csv";
    std::string notes;
    size_t threads = 1;
};

BenchOptions parse_bench_args(int argc, char** argv) {
    BenchOptions opts;
    bool have_model = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (i + 1 >= argc) usage_error("missing value for " + std::string(arg));
            return argv[++i];
        };

        if (arg == "--model") { opts.model_dir = next(); have_model = true; }
        else if (arg == "--config") { opts.config_name = std::string(next()); }
        else if (arg == "--out") { opts.out_path = std::string(next()); }
        else if (arg == "--notes") { opts.notes = std::string(next()); }
        else if (arg == "--threads") { opts.threads = parse_size(arg, next()); }
        else usage_error("unknown flag '" + std::string(arg) + "'");
    }

    if (!have_model) usage_error("--model is required");
    if (opts.threads != 1) usage_error("--threads must be 1 for now");

    return opts;
}

void run_bench(const BenchOptions& opts) {
    require_release_build();

    const Tokenizer tokenizer(opts.model_dir + "/tokenizer.json");
    const std::vector<int> prompt_ids = tokenizer.encode(kPrompt);

    const size_t max_seq = prompt_ids.size() + kDecodeTokens + 8;
    LoadedModel lm = load_model(opts.model_dir, max_seq);

    const Result result = measure(lm.model, prompt_ids);

    std::printf("prefill %.2f tok/s (median of %zu)\n", result.prefill_tps, kMeasuredRuns);
    std::printf("decode %.2f tok/s (median of %zu)\n", result.decode_tps, kMeasuredRuns);
    std::printf("peak RSS %.0f MB\n", result.peak_rss_mb);

    append_csv_row(opts.out_path, Row{
        current_git_hash(), current_date(), opts.config_name, opts.threads,
        result.prefill_tps, result.decode_tps, result.peak_rss_mb, opts.notes
    });

    std::printf("appended to %s\n", opts.out_path.c_str());
}

}  // namespace

void run(int argc, char** argv) {
    run_bench(parse_bench_args(argc, argv));
}

}  // namespace bench
