#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "mmap_file.h"

struct SafeTensor {
    std::string dtype;
    std::vector<size_t> shape;
    size_t start;
    size_t end;
};

struct SafeTensorModel {
    MmapFile mmap_file;
    std::unordered_map<std::string, SafeTensor> tensors;
    std::unordered_map<std::string, std::string> metadata;

    SafeTensorModel(const std::string& filepath): mmap_file{filepath}{}
};