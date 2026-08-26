#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <algorithm>

#include "loader/config.h"
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
                "  parity     check logits against the HF\n"
            );
}

std::size_t argmax(std::span<const float> data) {
    if (data.empty()) {
        return -1; // Or handle empty span as needed
    }
    
    auto max_it = std::max_element(data.begin(), data.end());
    return std::distance(data.begin(), max_it);
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    const char* cmd = argv[1];
    const int max_seq { 125 };

    // Loading the config, weights, model, and tokenizer
    ModelConfig config = load_config("models/qwen2.5-0.5b-instruct/config.json");
    Tokenizer tokenizer { "models/qwen2.5-0.5b-instruct/tokenizer.json" };
    SafeTensorModel safetensors {
        load_safetensors("models/qwen2.5-0.5b-instruct/model.safetensors")
    };
    
    Weights weights { std::move(safetensors), config };
    Model model { std::move(weights), config, max_seq };

    if (std::strcmp(cmd, "parity") == 0) {
    } else if (std::strcmp(cmd, "generate") == 0) {
        std::string prompt { argv[2] };
        if (argc < 3) {
            std::cout << "Prompt: ";
            std::getline(std::cin >> std::ws, prompt);
        }
        
        size_t generation_size;
        std::cout << "Max Generation size: ";
        if (!(std::cin >> generation_size)) {
            std::fprintf(stderr, "inferno: generation size must be a non-negative integer\n");
            return 1;
        }

        std::vector<int> ids = tokenizer.encode(prompt);

        for (size_t step = 0; step < generation_size; ++step) {
            std::span<const float> logits = model.forward(ids);
        
            const size_t next_id = argmax(logits);
            ids.push_back(static_cast<int>(next_id));
        
            if (static_cast<int>(next_id) == config.eos_token_id) {
                break;
            }
        }

        std::cout << tokenizer.decode(ids) << '\n';
    } else {
        std::fprintf(stderr, "inferno: unknown command '%s'\n\n", cmd);
        return 1;
    }

    return 0;
}
