#pragma once
#include <cassert>
#include <ostream>
#include "matrix.hpp"
#include "utils.hpp"
#include "ops.hpp"

struct OneLayer
{
    float learning_rate;
    Linear<float> L1;
    Linear<float> L2;
    Relu<float> act;
    SoftmaxCrossEntropy<float> loss_fct;

    OneLayer(const float lr, const uint32_t num_features, const uint32_t num_units, const uint32_t num_classes, const int seed = 0, const float dev = 0.01f)
        : L1(num_features, num_units, seed, dev), L2(num_units, num_classes, seed, dev)
    {
        learning_rate = lr;
    }

    Matrix<float> forward(Matrix<float> &X)
    {
        auto h1 = L1.forward(X);
        auto a1 = act.forward(h1);
        auto h2 = L2.forward(a1);

        auto Y_hat = loss_fct.forward_softmax(h2);
        return Y_hat;
    }

    float step(Matrix<float> &X, Matrix<float> &Y)
    {
        // forward-pass
        auto h1 = L1.forward(X);
        auto a1 = act.forward(h1);
        auto h2 = L2.forward(a1);
        auto Y_hat = loss_fct.forward_softmax(h2);
        auto loss = loss_fct.forward_cross_entropy(Y_hat, Y);

        // backward
        auto dLdh2 = loss_fct.backward(Y_hat, Y);
        auto dLdW2 = L2.backward_last_term(a1) * dLdh2;

        auto dh2da1 = L2.backward();
        auto da1dh1 = act.backward(h1);
        auto dh1dW1 = L1.backward_last_term(X);
        auto dLdW1 = dh1dW1 * (da1dh1.dot(dLdh2 * dh2da1));

        // Optimizer step
        L2.sgd_update(dLdW2, learning_rate);
        L1.sgd_update(dLdW1, learning_rate);

        return loss / X.rows;
    }
};
