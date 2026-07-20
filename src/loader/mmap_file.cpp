#include <iostream>
#include <filesystem>
#include <span>
#include <utility>
#include "mmap_file.h"

int map_file()
{
    const std::filesystem::path file_path = "src/loader/test.txt";
    MmapFile mmapfile { file_path };
    const char* data { mmapfile.get_data() };
    std::span<const std::byte> { mmapfile.get_bytes() };

    /* Now do something with the information. */
    for (std::size_t i = 0; i < mmapfile.get_size(); i++) {
        char c;

        c = data[i];
        std::cout << c;
    }

    std::cout << "\n";
    return 0;
}