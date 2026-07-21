#include <nlohmann/json.hpp>

#include "safetensors.h"

SafeTensorModel load_safetensors(const std::string& filepath) {
    SafeTensorModel model { filepath };

    // Get header size (from first 8 bytes) and then header
    uint64_t header_size = 0;
    std::memcpy(&header_size, model.mmap_file.get_data(), sizeof(uint64_t));
    
    const char* json_start = reinterpret_cast<const char*>(model.mmap_file.get_data() + 8);
    auto json_header = nlohmann::json::parse(json_start, json_start + header_size);

    model.data_start = 8 + header_size;

    // Populate a map with each tensor (in BF16)
    for (const auto& [tensor_name, tensor_info] : json_header.items()) {
        if (tensor_name == "__metadata__") {
            model.metadata = tensor_info.get<std::unordered_map<std::string, std::string>>();
            continue;
        }

        SafeTensor tensor;
        tensor.dtype = tensor_info["dtype"].get<std::string>();
        tensor.shape = tensor_info["shape"].get<std::vector<size_t>>();
        
        tensor.start = tensor_info["data_offsets"][0].get<size_t>();
        tensor.end = tensor_info["data_offsets"][1].get<size_t>();

        model.tensors[tensor_name] = tensor;
    }

    return model;
}