#pragma once

#include <span>

// y (size m) = weights (size m x n) * X (size n) + bias
// weights is a flattened m x n array with each row placed side by side
// bias is a size m list or a nullptr if there are none
void linear(std::span<float> y, std::span<const float> x, std::span<const float> weights, const float* bias);

// 
void rmsnorm(std::span<float> y, std::span<const float> x, std::span<const float> w, float eps);