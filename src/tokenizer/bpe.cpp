#include <limits>

#include "check.h"
#include "tokenizer/bpe.h"

namespace {

// returns left + ' ' + right
// space used to differentiate since spaces won't appear in left/right (replaced with G dot)
std::string make_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key += left;
    key += ' '; 
    key += right;
    return key;
}

// Split the chars from encode_bytes based on byte size (should be 1-2 bytes)
std::vector<std::string> split_characters(std::string_view mapped_chunk) {
    std::vector<std::string> characters;
    characters.reserve(mapped_chunk.size());

    for (size_t i = 0; i < mapped_chunk.size(); ) {
        size_t len = 1;

        // get the full char
        // (should be at most 2 chars, so an if can work, but this is incase a malformed input)
        while (i + len < mapped_chunk.size() && (static_cast<unsigned char>(mapped_chunk[i + len]) & 0xC0) == 0x80) {
            ++len; 
        }

        characters.emplace_back(mapped_chunk.substr(i, len)); 
        i += len;
    }

    return characters;
}

}  // namespace

void MergeTable::add(std::string_view left, std::string_view right, int rank) {
    const auto [it, inserted] = ranks_.emplace(make_key(left, right), rank);
    INFERNO_CHECK(inserted, "MergeTable.add: duplicate merge");
}

std::optional<int> MergeTable::rank(std::string_view left, std::string_view right) const {
    const auto found = ranks_.find(make_key(left, right));
    if (found == ranks_.end()) return std::nullopt; 

    return found->second;
}

std::vector<std::string> MergeTable::apply(std::string_view mapped_chunk) const {
    std::vector<std::string> pieces = split_characters(mapped_chunk);

    // Merge the lowest-ranked adjacent pair in order left to right
    // TODO O(n^2), O(nlog(m)) when I'm not lazy
    while (pieces.size() > 1) {
        int best_rank = std::numeric_limits<int>::max(); 
        size_t best = pieces.size(); 

        for (size_t i = 0; i + 1 < pieces.size(); ++i) { // sliding window of 2 to find best ranked merge
            const std::optional<int> r = rank(pieces[i], pieces[i + 1]);
            if (r && *r < best_rank) { 
                best_rank = *r;
                best = i; 
            }
        }

        if (best == pieces.size()) break;

        pieces[best] += pieces[best + 1];
        pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }

    return pieces;
}
