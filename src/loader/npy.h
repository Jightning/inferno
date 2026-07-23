#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class NpyDType { F32, I64 };

struct NpyArray {
    NpyDType dtype;
    std::vector<size_t> shape;
    std::vector<float> f32;
    std::vector<int64_t> i64;

    size_t numel() const;
};

NpyArray load_npy(const std::string& path);

std::vector<int64_t> load_npy_i64(const std::string& path);
std::vector<float> load_npy_f32(const std::string& path);