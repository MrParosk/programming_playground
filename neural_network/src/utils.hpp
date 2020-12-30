#pragma once
#include <random>
#include <vector>
#include <cstdint>
#include <math.h>
#include "matrix.hpp"

template <class T>
void fill_random(Matrix<T> &m, const int seed_num, const T dev)
{
    std::default_random_engine seed(seed_num);
    std::normal_distribution<T> generator(0, 1); // Sample from zero mean and unit variance

    std::vector<T> data;
    data.reserve(m.rows * m.cols);
    for (uint32_t i = 0; i < m.rows * m.cols; i++)
    {
        data.push_back(dev * generator(seed));
    }

    m.fill_vector(data);
}

template <class T>
constexpr Matrix<T> exponential(const Matrix<T> &m)
{
    auto return_matrix = m;
    for (uint32_t j = 0; j < return_matrix.cols; j++)
    {
        for (uint32_t i = 0; i < return_matrix.rows; i++)
        {
            return_matrix(i, j) = exp(return_matrix.get(i, j));
        }
    }
    return return_matrix;
}

template <class T>
constexpr Matrix<T> log(const Matrix<T> &m, T eps = 1e-8f)
{
    auto return_matrix = m;
    for (uint32_t j = 0; j < return_matrix.cols; j++)
    {
        for (uint32_t i = 0; i < return_matrix.rows; i++)
        {
            return_matrix(i, j) = log(return_matrix.get(i, j) + eps);
        }
    }
    return return_matrix;
}
