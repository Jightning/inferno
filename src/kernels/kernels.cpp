#include <cmath>
#include <algorithm>

#include "kernels/kernels.h"
#include "check.h"

void linear(
    std::span<float> y, 
    std::span<const float> x, 
    std::span<const float> weights, // weights is a flat version of the matrix (m x n)
    const float* bias // not a span so nullptr can be passed for no bias
) {
    //             m x n        n           m
    INFERNO_CHECK(weights.size() == x.size() * y.size(), "linear: dims of flattened weights doesn't match x and y");

    const size_t rows = y.size();
    const size_t cols = x.size();
    const float* xp = x.data();

    for (size_t m = 0; m < rows; ++m) {
        const float* weights_m = weights.data() + m * cols; // pointer to the start of the mth row
        float acc = bias != nullptr ? bias[m] : 0.0f; // accumulator to do work in a register (faster)

        for (size_t n = 0; n < cols; ++n) acc += xp[n] * weights_m[n];
        y[m] = acc;
    }
}

void rmsnorm(
    std::span<float> y, 
    std::span<const float> x, 
    std::span<const float> weights, 
    float eps
) {
    INFERNO_CHECK(weights.size() == x.size() && x.size() == y.size(), "rmsnorm: x, y, W all need to have the same size");
    
    size_t n = x.size();

    float sum_squares { 0.0f };
    for (size_t i = 0; i < n; ++i) {
        sum_squares += x[i] * x[i];
    }

    float inv { 1 / std::sqrt(sum_squares / n + eps) };
    for (size_t i = 0; i < n; ++i) {
        y[i] = weights[i] * x[i] * inv;
    }
}

void softmax(std::span<float> x) {
    if (x.empty()) { return; }
    size_t n = x.size();

    float maxx { x[0] };
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > maxx) maxx = x[i];
    }

    float sum_exp { 0.0f };
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maxx);
        sum_exp += x[i];
    }

    float inv_sum { 1.0f / sum_exp };
    for (size_t i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

void rope(std::span<float> vec, size_t pos, float theta) {
    /*
        for j in 0 .. 31:
            angle = pos * theta^(-2j / 64)
            out[j] = x[j] * cos(angle) - x[j+32] * sin(angle)
            out[j+32] = x[j+32] * cos(angle) + x[j] * sin(angle)
    */

    for (size_t i = 0; i < 31; ++i) {
        
    }
}