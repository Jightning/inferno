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
    INFERNO_CHECK(vec.size() % 2 == 0, "rope: vector size must be even");

    const size_t half = vec.size() / 2;
    const float dim = static_cast<float>(vec.size());

    for (size_t i = 0; i < half; ++i) {
        float angle { pos * std::pow(theta, -2.0f * static_cast<float>(i) / dim) };
        float cos_a { std::cos(angle) };
        float sin_a { std::sin(angle) };
        float v0 { vec[i] };
        float v1 { vec[i + half] };

        vec[i] = v0 * cos_a - v1 * sin_a;
        vec[i + half] = v1 * cos_a + v0 * sin_a;
    }
}

void silu_mul(std::span<float> gate, std::span<const float> up) {
    INFERNO_CHECK(gate.size() == up.size(), "silu_mul: gate and up must have the same size");

    size_t n = gate.size();
    for (size_t i = 0; i < n; ++i) {
        float z = gate[i];
        gate[i] = z / (1.0f + std::exp(-z)) * up[i];
    }
}

size_t argmax(std::span<const float> x) {
    INFERNO_CHECK(!x.empty(), "argmax: x must not be empty");

    size_t maxx = 0;
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] > x[maxx]) maxx = i;
    }

    // for nan stuff
    INFERNO_CHECK(std::isfinite(x[maxx]), "argmax: best element is not finite");
    return maxx;
}