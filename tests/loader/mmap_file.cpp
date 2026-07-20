#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>

#include "loader/mmap_file.h"

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& text) {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() / ("inferno_test_mmap_" + std::to_string(counter++) + ".bin");
        std::ofstream out(path_, std::ios::binary);
        out << text;
    }

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

const std::string kContents = "Hello, Memory Mapping";

}

TEST_CASE("MmapFile exposes a file's contents") {
    TempFile file(kContents);
    MmapFile mapped(file.path());

    CHECK(mapped.get_size() == kContents.size());
    CHECK(std::string(mapped.get_data(), mapped.get_size()) == kContents);
}

TEST_CASE("MmapFile exposes the same bytes through get_bytes") {
    TempFile file(kContents);
    MmapFile mapped(file.path());

    std::span<const std::byte> bytes = mapped.get_bytes();
    REQUIRE(bytes.size() == kContents.size());
    CHECK(std::memcmp(bytes.data(), kContents.data(), bytes.size()) == 0);
}

TEST_CASE("MmapFile maps binary data containing NUL bytes") {
    const std::string binary("ab\0cd", 5);
    TempFile file(binary);
    MmapFile mapped(file.path());

    REQUIRE(mapped.get_size() == 5);
    CHECK(std::memcmp(mapped.get_data(), binary.data(), 5) == 0);
}

TEST_CASE("MmapFile move construction transfers ownership") {
    TempFile file(kContents);

    MmapFile source(file.path());
    MmapFile destination(std::move(source));

    CHECK(std::string(destination.get_data(), destination.get_size()) == kContents);

    CHECK(source.get_size() == 0);
}


TEST_CASE("MmapFile rejects a file that cannot be opened") {
    std::filesystem::path missing = std::filesystem::temp_directory_path() / "inferno_no_such_mmap_file.bin";

    CHECK_THROWS_WITH_AS(MmapFile{missing}, "Failed to open file", std::runtime_error);
}

TEST_CASE("MmapFile rejects an empty file") {
    TempFile empty("");

    CHECK_THROWS_WITH_AS(MmapFile{empty.path()}, "Cannot map empty file", std::runtime_error);
}
