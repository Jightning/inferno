#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class MergeTable {
private:
    // Ranks for all the merges (lower = better) from tokenizer.json 
    std::unordered_map<std::string, int> ranks_;
public:
    // Adds { left + ' ' + right: rank } into ranks_
    void add(std::string_view left, std::string_view right, int rank); 
    
    // Gets the rank for left + right
    // nullopt if the pair doesn't exist
    std::optional<int> rank(std::string_view left, std::string_view right) const; 

    // Chunks are bpe merged based on rank (lowest rank, left to right)
    // ex. A B C D -> AB C D -> ABC D (no merge for ABC and D, so ABC and D are returned)
    std::vector<std::string> apply(std::string_view mapped_chunk) const;

    size_t size() const { return ranks_.size(); } // TODO remove?
};
