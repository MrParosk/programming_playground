#pragma once
#include <cassert>
#include "matrix.hpp"
#include "utils.hpp"
#include "ops.hpp"

struct OneLayer
{
    Linear<float> L1;
    Linear<float> L2;
    Relu<float> act;
    SoftmaxCrossEntropy<float> loss_fct;

    OneLayer(const uint32_t num_features, const uint32_t num_units, const uint32_t num_classes, const int seed, const float dev)
        : L1(num_features, num_units, seed, dev), L2(num_units, num_classes, seed, dev)
    {
    }

    float forward(Matrix<float> &X, Matrix<float> &Y)
    {
        auto h1 = L1.forward(X);
        auto a1 = act.forward(h1);
        auto h2 = L2.forward(a1);

        auto Y_hat = loss_fct.forward_softmax(h2);
        auto loss = loss_fct.forward_cross_entropy(Y_hat, Y);
        return loss;
    }
};
