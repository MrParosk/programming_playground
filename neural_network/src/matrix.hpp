#pragma once
#include <math.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>

template <class T>
struct Matrix
{
    std::vector<T> data;
    uint32_t rows;
    uint32_t cols;

    Matrix() : Matrix(0, 0) {}

    Matrix(const uint32_t num_rows, const uint32_t num_cols) : data(num_rows * num_cols, T(0.0))
    {
        rows = num_rows;
        cols = num_cols;
    }

    ~Matrix()
    {
        rows = 0;
        cols = 0;
        data.resize(0);
    }

    T &operator()(const uint32_t row_idx, const uint32_t col_idx)
    {
        // used for setting the value, set method
        if (row_idx >= rows && col_idx >= cols)
        {
            throw std::runtime_error("out of range of operator()");
        }
        return data[row_idx + col_idx * rows];
    }

    T get(const uint32_t row_idx, const uint32_t col_idx) const
    {
        if (row_idx >= rows && col_idx >= cols)
        {
            throw std::runtime_error("out of range of get");
        }

        return data[row_idx + col_idx * rows];
    }

    Matrix<T> &operator=(const Matrix<T> &other)
    {
        rows = other.rows;
        cols = other.cols;
        data.resize(rows * cols);

        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++)
            {
                (*this)(i, j) = other.get(i, j);
            }
        }

        return *this;
    }

    bool operator==(const Matrix<T> &other) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            return false;
        }

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                if (fabs(get(i, j) - other.get(i, j)) > 1e-6)
                {
                    return false;
                }
            }
        }

        return true;
    }

    Matrix<T> operator+(const Matrix<T> &other) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::runtime_error("Shapes incorrect in operator+");
        }

        Matrix<T> return_matrix(rows, cols);

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                return_matrix(i, j) = get(i, j) + other.get(i, j);
            }
        }

        return return_matrix;
    }

    Matrix<T> operator+(const T scalar) const
    {
        Matrix<T> return_matrix(rows, cols);

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                return_matrix(i, j) = get(i, j) + scalar;
            }
        }

        return return_matrix;
    }

    Matrix<T> operator-(const Matrix<T> &other) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::runtime_error("Shapes incorrect in operator-");
        }

        Matrix<T> return_matrix(rows, cols);

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                return_matrix(i, j) = get(i, j) - other.get(i, j);
            }
        }

        return return_matrix;
    }

    Matrix<T> operator-(const T scalar) const
    {
        Matrix<T> return_matrix(rows, cols);

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                return_matrix(i, j) = get(i, j) - scalar;
            }
        }

        return return_matrix;
    }

    Matrix<T> operator*(const Matrix<T> &other) const
    {
        if (cols != other.rows)
        {
            throw std::runtime_error("Shapes incorrect in operator*");
        }

        Matrix<T> return_matrix(rows, other.cols);

        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < other.cols; j++)
            {
                T temp_sum = 0;
                for (uint32_t k = 0; k < cols; k++)
                {
                    temp_sum += get(i, k) * other.get(k, j);
                }
                return_matrix(i, j) = temp_sum;
            }
        }

        return return_matrix;
    }

    Matrix<T> operator*(const T scalar) const
    {
        Matrix<T> return_matrix(rows, cols);

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                return_matrix(i, j) = get(i, j) * scalar;
            }
        }

        return return_matrix;
    }

    Matrix<T> operator/(const Matrix<T> &other) const
    {
        if (cols != other.cols || rows != other.rows)
        {
            throw std::runtime_error("Shapes incorrect in operator/");
        }

        Matrix<T> return_matrix(rows, cols);

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                return_matrix(i, j) = get(i, j) / other.get(i, j);
            }
        }

        return return_matrix;
    }

    Matrix<T> transpose() const
    {
        Matrix<T> return_matrix(cols, rows);
        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++)
            {
                return_matrix(j, i) = get(i, j);
            }
        }

        return return_matrix;
    }

    T sum() const
    {
        T sum_value = (T)0.0;
        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                sum_value += get(i, j);
            }
        }
        return sum_value;
    }

    Matrix<T> sum_over_cols() const
    {
        Matrix<T> S(rows, 1);

        for (uint32_t i = 0; i < rows; i++)
        {
            T temp_sum = (T)0.0;

            for (uint32_t j = 0; j < cols; j++)
            {
                temp_sum += get(i, j);
            }

            S(i, 0) = temp_sum;
        }

        return S;
    }

    void fill_vector(const std::vector<T> input_data)
    {
        if (input_data.size() != (rows * cols))
        {
            throw std::runtime_error("fill_vector got an vector of different size than the matrix one");
        }

        for (uint32_t i = 0; i < rows * cols; i++)
        {
            data[i] = input_data[i];
        }
    }

    Matrix<T> copy() const
    {
        Matrix<T> return_matrix(rows, cols);

        for (uint32_t j = 0; j < cols; j++)
        {
            for (uint32_t i = 0; i < rows; i++)
            {
                return_matrix(i, j) = get(i, j);
            }
        }

        return return_matrix;
    }

    void print_matrix() const
    {
        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++)
            {
                std::cout << get(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void print_shape() const
    {
        std::cout << rows << ", " << cols << std::endl;
    }
};
