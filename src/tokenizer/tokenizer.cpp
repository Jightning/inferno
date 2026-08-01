#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.h"
#include "check.h"
#include "tokenizer/byte_unicode.h"
#include "tokenizer/pretokenize.h"

using json = nlohmann::json;

namespace {

// ModelConfig::vocab_size for Qwen2.5
// hardcoded since it's just an upperbound, not worth the extra effort
constexpr int kVocabSizeLimit = 151936;

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream tokenizer_file(path);
    INFERNO_CHECK(tokenizer_file.is_open(), "Tokenizer: could not open {}", path);

    json tokenizer_json;
    try {
        tokenizer_file >> tokenizer_json;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("Tokenizer: malformed JSON syntax: " + std::string(e.what()));
    }

    try {
        const json& model = tokenizer_json.at("model");
        const json& vocab = model.at("vocab"); 
        const json& merges = model.at("merges");
        const json& added_tokens = tokenizer_json.at("added_tokens");

        // Adding all of vocab and added_tokens to vocab_
        int max_id = -1;
        const auto record = [&](std::string piece, int id) {
            INFERNO_CHECK(0 <= id && id < kVocabSizeLimit, "Tokenizer: id {} for '{}' is outside [0, {})", id, piece, kVocabSizeLimit);

            const auto [it, inserted] = vocab_.emplace(std::move(piece), id); 
            INFERNO_CHECK(inserted, "Tokenizer: duplicate '{}' (ids {} and {})", it->first, it->second, id); 

            max_id = std::max(max_id, id);
        };

        for (const auto& [piece, id] : vocab.items()) { // adding vocab
            INFERNO_CHECK(id.is_number_integer(), "Tokenizer: '{}' does not have an integer id", piece); 
            record(piece, id.get<int>());
        }

        for (const json& token : added_tokens) { // adding added_tokens
            auto piece = token.at("content").get<std::string>();
            const int id = token.at("id").get<int>();

            record(piece, id);
            const auto [it, inserted] = special_.emplace(std::move(piece), id); 
            INFERNO_CHECK(inserted, "Tokenizer: duplicate added_token '{}'", it->first);
        }

        // Filling inverse of vocab (id_to_piece)
        id_to_piece_.resize(static_cast<size_t>(max_id) + 1);
        std::vector<bool> assigned(id_to_piece_.size(), false);
         
        for (const auto& [piece, id] : vocab_) {
            id_to_piece_[static_cast<size_t>(id)] = piece;
            assigned[static_cast<size_t>(id)] = true;
        }

        for (size_t id = 0; id < assigned.size(); ++id) { // make sure no ids are skipped
            INFERNO_CHECK(assigned[id], "Tokenizer: nothing assigned to id {}", id, max_id);
        }

        // Adding merges to bpe MergeTable
        for (size_t rank = 0; rank < merges.size(); ++rank) {
            const json& merge = merges[rank];
            const auto joined = merge.get<std::string>();
             
            const size_t space = joined.find(' ');
            INFERNO_CHECK(space != std::string::npos,
                        "Tokenizer: merge at rank {} has no space: '{}'", rank, joined);
            
            const std::string left = joined.substr(0, space);
            const std::string right = joined.substr(space + 1);
            INFERNO_CHECK(vocab_.contains(left) && vocab_.contains(right),
                        "Tokenizer: merge at rank {} ('{}') has a half that is not in the vocab", rank, joined);

            merges_.add(left, right, static_cast<int>(rank));
        } 
    } catch (const json::out_of_range& e) {
        throw std::runtime_error("Tokenizer: a required key is missing: " + std::string(e.what()));
    } catch (const json::type_error& e) { 
        throw std::runtime_error("Tokenizer: type mismatch: " + std::string(e.what()));
    }
}

void Tokenizer::encode_ordinary(std::string_view text, std::vector<int>& ids) const {
    for (const std::string_view chunk : pretokenize(text)) {
        for (const std::string& piece : merges_.apply(encode_bytes(chunk))) {
            const auto found = vocab_.find(piece);
            
            INFERNO_CHECK(found != vocab_.end(), "Tokenizer.encode: '{}' is not in the vocab", piece);
            ids.push_back(found->second);
        } 
    }
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
    std::vector<int> ids;

    size_t pos = 0;
    while (pos <= text.size()) { // O(n * s)
        size_t special_at = text.size();
        size_t special_len = 0;
        int special_id = -1; 

        // finding special piece with nearest special char or by biggest size for consistency
        for (const auto& [piece, id] : special_) {
            const size_t found = text.find(piece, pos);
            
            if (found < special_at || (found == special_at && piece.size() > special_len)) {
                special_at = found; 
                special_len = piece.size();
                special_id = id;
            }
        }

        encode_ordinary(text.substr(pos, special_at - pos), ids); 
        if (special_len == 0) break; 
        
        ids.push_back(special_id);
        pos = special_at + special_len; 
    }

    return ids;
}

std::string Tokenizer::decode(std::span<const int> ids) const {
    std::string mapped;
    for (const int id : ids) mapped += id_to_piece(id);

    return decode_bytes(mapped);
}

std::optional<int> Tokenizer::piece_to_id(std::string_view piece) const {
    const auto found = vocab_.find(std::string(piece));

    if (found == vocab_.end()) return std::nullopt;
    return found->second;
}

std::string_view Tokenizer::id_to_piece(int id) const {
    INFERNO_CHECK(id >= 0 && static_cast<size_t>(id) < id_to_piece_.size(), "Tokenizer: no piece for id {}", id);
    return id_to_piece_[static_cast<size_t>(id)];
}
