#pragma once
#include "matrix.hpp"

template <class T>
struct Linear
{
    Matrix<T> W;

    Linear(const uint32_t num_features, const uint32_t num_units, const int seed, const float dev) : W(num_features, num_units)
    {
        fill_random(W, seed, dev);
    }

    Matrix<T> forward(const Matrix<T> &a) const
    {

        return a * W;
    }

    Matrix<T> backward(const Matrix<T> &a) const
    {
        return W;
    }

    void sgd_update(const Matrix<T> &dLdW, const T learning_rate)
    {
        W = W - (dLdW * learning_rate);
    }
};

template <class T>
struct Relu
{
    Relu() {}

    Matrix<T> forward(const Matrix<T> &m) const
    {
        auto return_matrix = m;
        for (uint32_t j = 0; j < return_matrix.cols; j++)
        {
            for (uint32_t i = 0; i < return_matrix.rows; i++)
            {
                if (return_matrix(i, j) < 0)
                {
                    return_matrix(i, j) = (T)0.0;
                }
            }
        }
        return return_matrix;
    }

    Matrix<T> backward(const Matrix<T> &m) const
    {
        auto return_matrix = m;
        for (uint32_t j = 0; j < return_matrix.cols; j++)
        {
            for (uint32_t i = 0; i < return_matrix.rows; i++)
            {
                if (return_matrix(i, j) < 0)
                {
                    return_matrix(i, j) = (T)0.0;
                }
                else
                {
                    return_matrix(i, j) = (T)1.0;
                }
            }
        }
        return return_matrix;
    }
};

template <class T>
struct SoftmaxCrossEntropy
{
    SoftmaxCrossEntropy() {}

    T forward_cross_entropy(const Matrix<T> &Y_hat, const Matrix<T> &Y) const
    {
        return ((Y.transpose() * log(Y_hat)) * ((T)-1.0)).sum();
    }

    Matrix<T> forward_softmax(const Matrix<T> &h) const
    {
        auto Y_hat = exponential(h);
        auto S = Y_hat.sum_over_cols();

        for (uint32_t j = 0; j < Y_hat.cols; j++)
        {
            for (uint32_t i = 0; i < Y_hat.rows; i++)
            {
                Y_hat(i, j) = Y_hat(i, j) / S(i, 0);
            }
        }

        return Y_hat;
    }

    Matrix<T> backward(const Matrix<T> &Y_hat, const Matrix<T> &Y) const
    {
        return Y - Y_hat;
    }
};
