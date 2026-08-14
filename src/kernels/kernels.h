#pragma once

#include <span>

// y (size m) = weights (size m x n) * x (size n) + bias
// weights is a flattened m x n array with each row placed side by side
// bias is a size m list or a nullptr if there are none
// sizes come from the spans and are checked; y must not alias x or weights
void linear(std::span<float> y, std::span<const float> x, std::span<const float> weights, const float* bias);

// y_i = weights_i * x_i / sqrt(mean(x^2) + eps)
void rmsnorm(std::span<float> y, std::span<const float> x, std::span<const float> weights, float eps);

// x_i = exp(x_i - max) / sum(exp(x - max))
void softmax(std::span<float> x); 

// Rotate vec based on pos and theta
// For positional encoding, vec size must be even
void rope(std::span<float> vec, size_t pos, float theta);

// gate_i = silu(gate_i) * up_i, where silu(z) = z / (1 + e^-z)
void silu_mul(std::span<float> gate, std::span<const float> up);

// Index of max
size_t argmax(std::span<const float> x);