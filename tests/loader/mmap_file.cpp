#include <doctest/doctest.h>

#include <fstream>
#include <filesystem>
#include <string>

#include "loader/mmap_file.h"

TEST_CASE("MMap basic functionality") {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::filesystem::path temp_file = temp_dir / "test_mmap_file.txt";

    std::string expected_data = "Hello, Memory Mapping!";
    {
        std::ofstream out(temp_file, std::ios::binary);
        out << expected_data;
    }

    {
        MmapFile mmapped_file(temp_file);

        std::string actual_data(mmapped_file.get_data(), mmapped_file.get_size());
        CHECK(actual_data == expected_data);
    }

    std::filesystem::path fake_path = temp_dir / "this_file_does_not_exist.xyz";
    CHECK_THROWS(MmapFile(fake_path));

    {
        MmapFile source_file(temp_file);
        MmapFile destination_file(std::move(source_file));

        std::string moved_data(destination_file.get_data(), destination_file.get_size());
        CHECK(moved_data == expected_data);

        CHECK(source_file.get_size() == 0);
    }

    std::filesystem::remove(temp_file);
}