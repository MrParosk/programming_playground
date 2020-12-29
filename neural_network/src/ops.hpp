#pragma once
#include "matrix.hpp"

template <class T>
struct CrossEntropy
{
    CrossEntropy() {}

    T forward(const Matrix<T> &Y_hat, const Matrix<T> &Y) const
    {
        return ((Y.transpose() * log(Y_hat)) * ((T)-1.0)).sum();
    }

    Matrix<T> backward(const Matrix<T> &Y_hat, const Matrix<T> &Y) const
    {
        return (Y / Y_hat - ((Y - 1) / (Y_hat - 1))) * -(T)1.0;
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
