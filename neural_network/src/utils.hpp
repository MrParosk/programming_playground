#pragma once
#include <random>
#include <vector>
#include <cstdint>
#include <tuple>
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

template <class T>
constexpr float accuracy(const Matrix<T> &Y, const Matrix<T> &Y_hat)
{
    const T tol = (T)1e-6;
    float num_correct = 0.0f;

    for (uint32_t i = 0; i < Y.rows; i++)
    {
        T max_val = (T)-1.0;
        uint32_t max_idx = 0;

        for (uint32_t j = 0; j < Y.cols; j++)
        {
            if (Y_hat.get(i, j) > max_val)
            {
                max_val = Y_hat.get(i, j);
                max_idx = j;
            }
        }

        if (fabs(Y.get(i, max_idx) - (T)1.0) < tol)
        {
            num_correct++;
        }
    }

    return num_correct / ((float)Y.rows);
}

template <class T>
std::tuple<Matrix<T>, Matrix<T>> generate_data(const std::vector<std::vector<T>> &center_means,
                                               const uint32_t num_samples_per_class, const uint32_t num_features, const uint32_t num_classes,
                                               const T dev = 0.1, const uint32_t seed_num = 0)
{
    std::default_random_engine seed(seed_num);
    std::normal_distribution<T> generator(0, 1); // Sample from zero mean and unit variance

    // first col of X is the bias term
    Matrix<T> X(num_classes * num_samples_per_class, num_features + 1);
    Matrix<T> Y(num_classes * num_samples_per_class, num_classes);

    for (uint32_t c = 0; c < num_classes; c++)
    {
        auto center = center_means[c];

        if (center.size() != num_features)
        {
            throw std::runtime_error("center-size is not the same as num_features");
        }

        auto offset = c * num_samples_per_class;
        for (uint32_t i = 0; i < num_samples_per_class; i++)
        {
            Y(offset + i, c) = (T)1.0;
            X(offset + i, 0) = (T)1.0;
            for (uint32_t j = 1; j < num_features; j++)
            {
                X(offset + i, j) = dev * generator(seed) + center[j];
            }
        }
    }

    return std::make_tuple(X, Y);
}
