#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tokenizer/bpe.h"

class Tokenizer {
private:
    std::unordered_map<std::string, int> vocab_; // any piece -> id
    std::unordered_map<std::string, int> special_; // added_tokens piece -> token
    std::vector<std::string> id_to_piece_; // id -> piece, dense
    
    MergeTable merges_;

    // encodes just the non-special_ text to ids
    void encode_ordinary(std::string_view text, std::vector<int>& ids) const;
public:
    // Adds in vocab and merges from path 
    // this structure: { model: { vocab: { "content": id, ...}, merges: [...] }, added_tokens: [ { content, id }, ... ] }
    explicit Tokenizer(const std::string& path);

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // text -> ids
    // pretokenize -> convert to utf -> apply bpe merges -> convert remaining pieces to ids
    std::vector<int> encode(std::string_view text) const;

    // ids -> text
    // id -> piece -> convert to byte
    std::string decode(std::span<const int> ids) const;

    std::optional<int> piece_to_id(std::string_view piece) const;
    std::string_view id_to_piece(int id) const;
    size_t size() const { return vocab_.size(); }
};
