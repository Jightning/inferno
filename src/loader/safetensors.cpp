#include "safetensors.h"
#include "mmap_file.h"

void load_safetensors(const std::string& filepath) {
    MmapFile file { filepath };

    uint64_t header_len = 0;
    std::unordered_map<std::string, SafeTensor> tensor_info;

    SafeTensor* header = reinterpret_cast<SafeTensor*>(file.get_data());

}