#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "loader/config.h"
#include "model/model.h"

namespace cli {

void print_usage();

[[noreturn]] void usage_error(const std::string& message);

struct LoadedModel {
    ModelConfig config;
    Model model;
};

LoadedModel load_model(const std::string& model_dir, size_t max_seq);

size_t parse_size(std::string_view flag, std::string_view value);

std::vector<int> parse_token_ids(std::string_view csv);

}  // namespace cli
