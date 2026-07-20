#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <nlohmann/json.hpp>

struct SafeTensor {
    std::string dtype;
    std::vector<size_t> shape;
    size_t offset_begin;
    size_t offset_end;
};