#pragma once

#include <format>
#include <stdexcept>

// macro for load time error handling
#define INFERNO_CHECK(cond, ...) do { if (!(cond)) throw std::runtime_error(std::format(__VA_ARGS__)); } while (0)
