#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>

#include "loader/npy.h"

const std::string PARITY_DATA_DIR = "/parity_data";

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& bytes) {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() / ("inferno_test_npy_" + std::to_string(counter++) + ".npy");
        std::ofstream out(path_, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

std::string make_npy(const std::string& dict, const std::string& payload) {
    std::string header = dict;
    const size_t unpadded = 10 + header.size() + 1;  // +1 for the newline
    header.append((64 - unpadded % 64) % 64, ' ');
    header += '\n';

    std::string out("\x93NUMPY\x01\x00", 8);
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.append(reinterpret_cast<const char*>(&header_len), sizeof header_len);
    out += header;
    out += payload;
    return out;
}

template <typename T>
std::string raw(std::initializer_list<T> values) {
    std::string out;
    for (const T value : values) out.append(reinterpret_cast<const char*>(&value), sizeof value);
    return out;
}

const std::string kI64Dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }";
const std::string kI64Payload = raw<int64_t>({7, -8, 9});

}  // namespace

TEST_CASE("load_npy reads an int64 array") {
    TempFile file(make_npy(kI64Dict, kI64Payload));
    const NpyArray array = load_npy(file.path());

    CHECK(array.dtype == NpyDType::I64);
    CHECK(array.shape == std::vector<size_t>{3});
    CHECK(array.numel() == 3);
    CHECK(array.f32.empty());
    REQUIRE(array.i64.size() == 3);
    CHECK(array.i64[0] == 7);
    CHECK(array.i64[1] == -8);
    CHECK(array.i64[2] == 9);
}

TEST_CASE("load_npy reads a 2-D float32 array in C order") {
    TempFile file(make_npy("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
                           raw<float>({1, 2, 3, 4, 5, 6})));
    const NpyArray array = load_npy(file.path());

    CHECK(array.dtype == NpyDType::F32);
    CHECK(array.shape == std::vector<size_t>{2, 3});
    CHECK(array.numel() == 6);
    CHECK(array.i64.empty());
    REQUIRE(array.f32.size() == 6);
    CHECK(array.f32[3] == 4.0f);
    CHECK(array.f32[5] == 6.0f);
}

TEST_CASE("load_npy handles the shapes numpy writes at the edges") {
    SUBCASE("a trailing comma is a separator, not a zero dimension") {
        TempFile file(make_npy(kI64Dict, kI64Payload));
        CHECK(load_npy(file.path()).shape == std::vector<size_t>{3});
    }
    SUBCASE("an empty tuple is a 0-d scalar holding one element") {
        TempFile file(make_npy("{'descr': '<i8', 'fortran_order': False, 'shape': (), }",
                               raw<int64_t>({42})));
        const NpyArray array = load_npy(file.path());
        CHECK(array.shape.empty());
        CHECK(array.numel() == 1);
        CHECK(array.i64[0] == 42);
    }
    SUBCASE("a zero dimension means no elements at all") {
        TempFile file(make_npy("{'descr': '<f4', 'fortran_order': False, 'shape': (0,), }", ""));
        const NpyArray array = load_npy(file.path());
        CHECK(array.shape == std::vector<size_t>{0});
        CHECK(array.numel() == 0);
        CHECK(array.f32.empty());
    }
    SUBCASE("more than two dimensions") {
        TempFile file(make_npy("{'descr': '<i8', 'fortran_order': False, 'shape': (2, 3, 4), }",
                               std::string(2 * 3 * 4 * sizeof(int64_t), '\0')));
        const NpyArray array = load_npy(file.path());
        CHECK(array.shape == std::vector<size_t>{2, 3, 4});
        CHECK(array.numel() == 24);
    }
}

TEST_CASE("load_npy accepts header keys in any order") {
    TempFile file(make_npy("{'shape': (3,), 'fortran_order': False, 'descr': '<i8', }", kI64Payload));
    const NpyArray array = load_npy(file.path());

    CHECK(array.dtype == NpyDType::I64);
    CHECK(array.shape == std::vector<size_t>{3});
    CHECK(array.i64[0] == 7);
}

TEST_CASE("load_npy rejects a file that is not a .npy") {
    SUBCASE("wrong magic") {
        std::string bytes = make_npy(kI64Dict, kI64Payload);
        bytes[4] = 'Z';
        TempFile file(bytes);
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }

    SUBCASE("shorter than the magic number") {
        TempFile file("\x93NU");
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }
    SUBCASE("missing file") {
        CHECK_THROWS_AS(load_npy("/nonexistent/inferno/does_not_exist.npy"), std::runtime_error);
    }
}

TEST_CASE("load_npy rejects a header format it cannot decode") {
    std::string bytes = make_npy(kI64Dict, kI64Payload);
    bytes[6] = 2;
    TempFile file(bytes);

    CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
}

TEST_CASE("load_npy rejects header contents it cannot honor") {
    SUBCASE("column-major data would invalidate every computed index") {
        TempFile file(make_npy("{'descr': '<i8', 'fortran_order': True, 'shape': (3,), }", kI64Payload));
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }
    SUBCASE("an unsupported dtype is named in the message") {
        TempFile file(make_npy("{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }",
                               std::string(24, '\0')));
        CHECK_THROWS_WITH_AS(load_npy(file.path()),
                             "Unexpected dtype '<f8', expected '<f4' or '<i8'", std::runtime_error);
    }
    SUBCASE("big-endian data is not byte-swapped, so it is refused") {
        TempFile file(make_npy("{'descr': '>i8', 'fortran_order': False, 'shape': (3,), }", kI64Payload));
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }
    SUBCASE("a missing key would otherwise default silently") {
        TempFile file(make_npy("{'descr': '<i8', 'fortran_order': False, }", kI64Payload));
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }
    SUBCASE("a dimension too large for size_t") {
        TempFile file(make_npy("{'descr': '<i8', 'fortran_order': False, 'shape': (99999999999999999999,), }", ""));
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }

    SUBCASE("an unterminated quote terminates the scan") {
        TempFile file(make_npy("{'descr': '<i8', 'fortran_order': False, 'shape: (3,), }", kI64Payload));
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }
}

TEST_CASE("load_npy rejects a payload whose size disagrees with the shape") {
    SUBCASE("truncated") {
        TempFile file(make_npy(kI64Dict, raw<int64_t>({7, -8})));
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }
    SUBCASE("over-long") {
        TempFile file(make_npy(kI64Dict, raw<int64_t>({7, -8, 9, 10})));
        CHECK_THROWS_AS(load_npy(file.path()), std::runtime_error);
    }
}

TEST_CASE("The typed loaders return the payload and reject the wrong dtype") {
    TempFile ints(make_npy(kI64Dict, kI64Payload));
    TempFile floats(make_npy("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }",
                             raw<float>({1.5f, 2.5f})));

    CHECK(load_npy_i64(ints.path()) == std::vector<int64_t>{7, -8, 9});
    CHECK(load_npy_f32(floats.path()) == std::vector<float>{1.5f, 2.5f});

    CHECK_THROWS_AS(load_npy_i64(floats.path()), std::runtime_error);
    CHECK_THROWS_AS(load_npy_f32(ints.path()), std::runtime_error);
}

TEST_CASE("load_npy reads the parity fixtures M3 checks against") {
    const std::filesystem::path dir = PARITY_DATA_DIR;
    const std::filesystem::path tokens = dir / "prompt00_tokens.npy";
    const std::filesystem::path logits = dir / "prompt00_logits.npy";

    if (!std::filesystem::exists(tokens) || !std::filesystem::exists(logits)) {
        MESSAGE("skipped: no parity data at " << dir.string());
        return;
    }

    const std::vector<int64_t> ids = load_npy_i64(tokens.string());
    REQUIRE(ids.size() == 133);
    CHECK(ids[0] == 785);
    CHECK(ids[1] == 6722);
    CHECK(ids.back() == 16807);

    const NpyArray scores = load_npy(logits.string());
    CHECK(scores.dtype == NpyDType::F32);
    CHECK(scores.shape == std::vector<size_t>{128, 151936});
    REQUIRE(scores.f32.size() == 128u * 151936u);
    CHECK(scores.f32[0] == doctest::Approx(7.548954f));
}
