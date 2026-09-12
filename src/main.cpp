#include <cstdio>
#include <exception>
#include <string_view>

#include "commands/args.h"
#include "commands/bench.h"
#include "commands/generate.h"
#include "commands/parity.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        cli::print_usage();
        return 1;
    }

    try {
        const std::string_view cmd = argv[1];
        if (cmd == "generate") { // predict next n tokens
            generate::run(argc, argv);
        } else if (cmd == "parity") { // comparison with Hugging Face (`/scripts/parity.py`)
            if (!parity::run(argc, argv)) return 1;
        } else if (cmd == "bench") { // benchmarking
            bench::run(argc, argv);
        } else if (cmd == "quantize") {
            // TODO gotta work on this
            std::fprintf(stderr, "inferno: '%s' is not yet implemented\n", argv[1]);
            return 1;
        } else {
            std::fprintf(stderr, "inferno: unknown command '%s'\n\n", argv[1]);
            cli::print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inferno: %s\n", e.what());
        return 1;
    }

    return 0;
}
