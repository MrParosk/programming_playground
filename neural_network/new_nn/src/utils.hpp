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
void exponential(Matrix<T> &m)
{
    for (uint32_t i = 0; i < m.rows; ++i)
    {
        for (uint32_t j = 0; j < m.cols; ++j)
        {
            m(i, j) = exp(m(i, j));
        }
    }
}

template <class T>
Matrix<T> softmax(Matrix<T> &m)
{
    auto P = m;
    exponential(P);
    auto S = P.sum_over_cols();

    for (uint32_t i = 0; i < P.rows; ++i)
    {
        for (uint32_t j = 0; j < P.cols; ++j)
        {
            P(i, j) = P(i, j) / S(i, 0);
        }
    }

    return P;
}
