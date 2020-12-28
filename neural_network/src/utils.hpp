#pragma once
#include <random>
#include <vector>
#include <cstdint>
#include <math.h>
#include "matrix.hpp"

template <class T>
void fill_random(Matrix<T> &m, const int seed_num, const float dev)
{
    std::default_random_engine seed(seed_num);
    std::normal_distribution<T> generator(0, 1); // Sample from zero mean and unit variance

    std::vector<T> data;
    data.reserve(m.rows * m.cols);
    for (uint32_t i = 0; i < m.rows * m.cols; ++i)
    {
        data.push_back(dev * generator(seed));
    }

    m.fill_vector(data);
}

template <class T>
constexpr void exponential(Matrix<T> &m)
{
    for (uint32_t j = 0; j < m.cols; ++j)
    {
        for (uint32_t i = 0; i < m.rows; ++i)
        {
            m(i, j) = exp(m(i, j));
        }
    }
}

template <class T>
constexpr Matrix<T> softmax(const Matrix<T> &m)
{
    auto P = m;
    exponential(P);
    auto S = P.sum_over_cols();

    for (uint32_t j = 0; j < P.cols; ++j)
    {
        for (uint32_t i = 0; i < P.rows; ++i)
        {
            P(i, j) = P(i, j) / S(i, 0);
        }
    }

    return P;
}

template <class T>
constexpr void log(Matrix<T> &m)
{
    for (uint32_t j = 0; j < m.cols; ++j)
    {
        for (uint32_t i = 0; i < m.rows; ++i)
        {
            m(i, j) = log(m(i, j));
        }
    }
}

template <class T>
constexpr Matrix<T> relu(Matrix<T> &m)
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
