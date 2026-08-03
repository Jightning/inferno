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

    for (size_t m = 0; m < y.size(); ++m) {
        const float* weights_m = weights.data() + m * x.size(); // pointer to the start of the mth row
        float acc = bias != nullptr ? bias[m] : 0.0f; // accumulator to do work in a register (faster)

        for (size_t n = 0; n < x.size(); ++n) acc += x[n] * weights_m[n];
        y[m] = acc;  
    }
}

// void rmsnorm(
//     std::span<float> y, 
//     std::span<const float> x, 
//     std::span<const float> w, 
//     float eps
// ) {
    
// }