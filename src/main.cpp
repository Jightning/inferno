#include <cstdio>
#include <cstring>
#include <iostream>
#include "loader/config.h"
#include "loader/mmap_file.h"
#include "loader/npy.h"
namespace {

void print_usage() {
    std::fprintf(stderr,
                 "usage: inferno <command> [options]\n"
                 "\n"
                 "commands:\n"
                 "  generate   generate tokens from a prompt\n"
                 "  bench      report prefill/decode tok/s and peak RSS\n"
                 "  parity     check logits against the HF reference dump\n");
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    const char* cmd = argv[1];
    if (std::strcmp(cmd, "parity") == 0) {
        ModelConfig config { load_config("models/qwen2.5-0.5b-instruct/config.json") };
        std::cout << "Config Loaded\n";
    } else if (std::strcmp(cmd, "test") == 0) {
        std::string path { "parity_data/prompt00_logits.npy" };
        load_npy(path);
    } else {
        std::fprintf(stderr, "inferno: unknown command '%s'\n\n", cmd);
        return 1;
    }

    return 0;
}
