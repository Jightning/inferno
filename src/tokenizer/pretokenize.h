#pragma once

#include <string_view>
#include <vector>

// Gets the the length of the tokens chunk based on the regex (found in tokenizer.json for qwen)
// using the qwen2.5 regex (might need to change if I chose another model, gotta confirm)
// custom regex since the regex library can't process some of this, plus it's apparently slow
size_t next_chunk_length(std::string_view text, size_t pos);

// Splits into chunks based on the regex defined at tokenizer.json
std::vector<std::string_view> pretokenize(std::string_view text); 
