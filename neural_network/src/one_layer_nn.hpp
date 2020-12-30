#pragma once
#include <cassert>
#include <ostream>
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

    void step(Matrix<float> &X, Matrix<float> &Y)
    {

        float lr = 0.1f;

        auto h1 = L1.forward(X);
        auto a1 = act.forward(h1);
        auto h2 = L2.forward(a1);

        auto Y_hat = loss_fct.forward_softmax(h2);

        Y_hat.print_matrix();
        auto loss = loss_fct.forward_cross_entropy(Y_hat, Y);

        auto dLdh2 = loss_fct.backward(Y_hat, Y);

        auto dLdW2 = L2.backward_last_term(a1) * dLdh2;

        auto dh2da1 = L2.backward();
        auto da1dh1 = act.backward(h1);
        auto dh1dW1 = L1.backward_last_term(X);
        //auto dLdW1 = dLdh2 * dh2da1 * da1dh1 * dh1dW1;

        L2.sgd_update(dLdW2, lr);
        // L2.sgd_update(dLdW1, lr);

        std::cout << loss << std::endl;
    }
};
