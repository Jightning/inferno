#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

class Model;

// Benchmarking
// Writes results to results.csv
namespace bench {

// things to keep constant across benchmarks
extern const char* const kPrompt; // ~64 tokens
constexpr size_t kDecodeTokens = 128; // tokens generated per measured run
constexpr size_t kWarmupDecodeTokens = 2;
constexpr size_t kMeasuredRuns = 3; // odd for median

constexpr const char* kCsvHeader = "git_hash,date,config,threads,prefill_tps,decode_tps,peak_rss_mb,notes";

struct Row {
    std::string git_hash;
    std::string date;
    std::string config;
    size_t threads;
    double prefill_tps;
    double decode_tps;
    double peak_rss_mb;
    std::string notes;
};

struct Result {
    double prefill_tps;
    double decode_tps;
    double peak_rss_mb;
};

struct Timings {
    double prefill_sec;
    double decode_sec;
};

// Tests the timing for the prefill (first forward) and the decoding (the rest)
Timings time_generation(Model& model, std::span<const int> prompt_ids, size_t decode_tokens);

void require_release_build();

// Gets the median timings across kMeasuredRuns
Result measure(Model& model, std::span<const int> prompt_ids);

double median(std::vector<double> values);

double peak_rss_mb();
std::string current_date();

// Short commit hash to know where the bench came from
std::string current_git_hash();

std::string format_csv_row(const Row& row);

void append_csv_row(const std::string& path, const Row& row);

// inferno bench --model <dir> [--config] [--out] [--notes] [--threads 1]
void run(int argc, char** argv);

}  // namespace bench
