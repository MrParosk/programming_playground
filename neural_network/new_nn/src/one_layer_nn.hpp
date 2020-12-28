#pragma once
#include <cassert>
#include "matrix.hpp"
#include "utils.hpp"

class OneLayer
{
public:
    Matrix<float> W1;
    Matrix<float> W2;

    OneLayer(const uint32_t num_features, const uint32_t num_units, const uint32_t num_classes, const int seed, const float dev) : W1(num_features, num_units), W2(num_units, num_classes)
    {
        fill_random(W1, seed, dev);
        fill_random(W2, seed, dev);
    }

    Matrix<float> forward(Matrix<float> &X)
    {
        auto h = X * W1;

        auto a = relu<float>(h);
        auto s = a * W2;
        auto P = softmax(s);
        return P;
    }
};
