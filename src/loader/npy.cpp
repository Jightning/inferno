#include <charconv>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>

#include "npy.h"
#include "check.h"

namespace {

size_t dtype_size(NpyDType dtype) {
    return dtype == NpyDType::F32 ? sizeof(float) : sizeof(int64_t);
}

struct NpyHeader {
    std::string descr;
    bool fortran_order = false;
    std::vector<size_t> shape;
};

// Get header values, ex. header:
// {'descr': '<i8', 'fortran_order': False, 'shape': (133,), }
NpyHeader parse_header(std::string_view header) {
    NpyHeader out;
    std::string_view key;
    bool seen_descr = false, seen_fortran = false, seen_shape = false;

    for (size_t i = 0; i < header.size(); ++i) {
        const char c = header[i];

        if (c == '\'') { // get the key
            const size_t end = header.find('\'', i + 1);
            // npos here would wrap i to 0 on the next ++i and restart the scan forever
            INFERNO_CHECK(end != std::string_view::npos, "Unterminated quote in header");
            const std::string_view value = header.substr(i + 1, end - i - 1);

            if (key == "descr") {
                out.descr = std::string(value);
                seen_descr = true;
                key = {};
            } else key = value;

            i = end;
        // get the value based on that key
        } else if (key == "fortran_order" && (c == 'T' || c == 'F')) {
            out.fortran_order = (c == 'T');
            seen_fortran = true;
            key = {};
        } else if (key == "shape" && c >= '0' && c <= '9') {
            size_t dim = 0;
            const auto [end, ec] = std::from_chars(header.data() + i, header.data() + header.size(), dim);
            INFERNO_CHECK(ec == std::errc{}, "Malformed shape dimension in header");

            out.shape.push_back(dim);
            i = static_cast<size_t>(end - header.data()) - 1;
        } else if (key == "shape" && c == ')') {
            seen_shape = true;
            key = {};
        }
    }

    INFERNO_CHECK(seen_descr && seen_fortran && seen_shape, "Header is missing one of descr/fortran_order/shape");
    return out;
}

// Fills the vector
template <typename T>
void read_payload(std::ifstream& file, std::vector<T>& out, size_t count) {
    out.resize(count);
    const std::streamsize nbytes = static_cast<std::streamsize>(count * sizeof(T));

    file.read(reinterpret_cast<char*>(out.data()), nbytes);
    INFERNO_CHECK(file.gcount() == nbytes, "Payload read returned {} bytes, expected {}", file.gcount(), nbytes);
}

} // namespace

size_t NpyArray::numel() const {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

NpyArray load_npy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    INFERNO_CHECK(file.is_open(), "{}: failed to open", path);

    // Making sure it's a numpy file of version 1.0
    const unsigned char numpy_magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    unsigned char magic[6];
    file.read(reinterpret_cast<char*>(magic), sizeof magic);
    INFERNO_CHECK(std::cmp_equal(file.gcount(), sizeof magic) && std::memcmp(magic, numpy_magic, sizeof magic) == 0, 
                "Bad magic number, not a numpy file");

    unsigned char version[2];
    file.read(reinterpret_cast<char*>(version), sizeof version);
    INFERNO_CHECK(std::cmp_equal(file.gcount(), sizeof version), "Couldn't get full version");
    INFERNO_CHECK(version[0] == 1 && version[1] == 0, "Numpy file must be version 1.0");

    // Reading header
    NpyArray npy_array;

    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), sizeof header_len);
    INFERNO_CHECK(std::cmp_equal(file.gcount(), sizeof header_len), "Couldn't get full header length");

    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);
    INFERNO_CHECK(std::cmp_equal(file.gcount(), header_len), "Couldn't get full header string");

    const NpyHeader header = parse_header(header_str);
    npy_array.shape = header.shape;

    INFERNO_CHECK(header.descr == "<f4" || header.descr == "<i8",
                "Unexpected dtype '{}', expected '<f4' or '<i8'", header.descr);
    npy_array.dtype = (header.descr == "<f4" ? NpyDType::F32 : NpyDType::I64);

    INFERNO_CHECK(!header.fortran_order, "fortran_order must be False");

    // Read rest of the content
    const std::streamoff data_start = file.tellg();
    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    file.seekg(data_start, std::ios::beg);
    
    // make sure bytes match up with size
    const size_t expected_bytes = npy_array.numel() * dtype_size(npy_array.dtype);
    const size_t actual_bytes = static_cast<size_t>(file_size - data_start);
    INFERNO_CHECK(actual_bytes == expected_bytes, "Shape needs {} bytes of data, the file holds {}", expected_bytes, actual_bytes);

    if (npy_array.dtype == NpyDType::F32) read_payload(file, npy_array.f32, npy_array.numel());
    else read_payload(file, npy_array.i64, npy_array.numel());

    return npy_array;
}

std::vector<int64_t> load_npy_i64(const std::string& path) {
    NpyArray a = load_npy(path);
    INFERNO_CHECK(a.dtype == NpyDType::I64, "{}: expected int64, got float32", path);
    return std::move(a.i64);
}

std::vector<float> load_npy_f32(const std::string& path) {
    NpyArray a = load_npy(path);
    INFERNO_CHECK(a.dtype == NpyDType::F32, "{}: expected float32, got int64", path);
    return std::move(a.f32);
}
