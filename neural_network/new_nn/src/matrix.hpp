#pragma once
#include <math.h>
#include <iostream>
#include <vector>
#include <cstdint>

template <class T>
class Matrix
{
public:
    std::vector<T> data;
    uint32_t cols;
    uint32_t rows;

    Matrix()
    {
        rows = 0;
        cols = 0;
        data = std::vector<T>();
        data.reserve(0);
    }

    Matrix(const uint32_t num_rows, const uint32_t num_cols)
    {
        rows = num_rows;
        cols = num_cols;
        uint32_t num_elements = num_rows * num_cols;
        data = std::vector<T>(num_elements, (T)0.0);
    }

    ~Matrix()
    {
        rows = 0;
        cols = 0;
        data.resize(0);
    }

    T &operator()(const uint32_t row_idx, const uint32_t col_idx)
    {
        return this->data[row_idx + col_idx * this->rows];
    }

    Matrix operator=(Matrix &other)
    {
        rows = other.rows;
        cols = other.cols;

        data.resize(0);
        data.resize(rows * cols);

        for (uint32_t i = 0; i < rows * cols; i++)
        {
            data[i] = other.data[i];
        }

        return *this;
    }

    bool operator==(Matrix &other)
    {
        if (this->rows != other.rows || this->cols != other.cols)
        {
            return false;
        }

        for (uint32_t i = 0; i < this->rows; i++)
        {
            for (uint32_t j = 0; j < this->cols; j++)
            {
                if (fabs((*this)(i, j) - other(i, j)) > 1e-6)
                {
                    return false;
                }
            }
        }

        return true;
    }

    Matrix operator+(Matrix &other)
    {
        if (this->rows != other.rows || this->cols != other.cols)
        {
            throw std::runtime_error("Shapes incorrect in operator+");
        }

        Matrix return_matrix(this->rows, this->cols);

        for (uint32_t i = 0; i < this->rows; i++)
        {
            for (uint32_t j = 0; j < this->cols; j++)
            {
                return_matrix(i, j) = (*this)(i, j) + other(i, j);
            }
        }

        return return_matrix;
    }

    Matrix operator-(Matrix &other)
    {
        if (this->rows != other.rows || this->cols != other.cols)
        {
            throw std::runtime_error("Shapes incorrect in operator-");
        }

        Matrix return_matrix(this->rows, this->cols);

        for (uint32_t i = 0; i < this->rows; i++)
        {
            for (uint32_t j = 0; j < this->cols; j++)
            {
                return_matrix(i, j) = (*this)(i, j) - other(i, j);
            }
        }

        return return_matrix;
    }

    Matrix operator*(Matrix &other)
    {
        if (this->cols != other.rows)
        {
            throw std::runtime_error("Shapes incorrect in operator*");
        }

        Matrix return_matrix(this->rows, other.cols);

        for (uint32_t i = 0; i < this->rows; ++i)
        {
            for (uint32_t j = 0; j < other.cols; ++j)
            {
                T temp_sum = 0;
                for (uint32_t k = 0; k < this->cols; k++)
                {
                    temp_sum += (*this)(i, k) * other(k, j);
                }
                return_matrix(i, j) = temp_sum;
            }
        }

        return return_matrix;
    }

    void fill_vector(std::vector<T> input_data)
    {
        if (input_data.size() != data.size())
        {
            throw std::runtime_error("fill_vector got an vector of different size than the matrix one");
        }

        for (uint32_t i = 0; i < data.size(); i++)
        {
            data[i] = input_data[i];
        }
    }

    void print_matrix()
    {
        for (uint32_t i = 0; i < rows; ++i)
        {
            for (uint32_t j = 0; j < cols; ++j)
            {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix<T> sum_over_cols()
    {
        Matrix<T> S(rows, 1);

        for (uint32_t i = 0; i < rows; ++i)
        {
            T temp_sum = (T)0.0;

            for (uint32_t j = 0; j < cols; ++j)
            {
                temp_sum += (*this)(i, j);
            }

            S(i, 0) = temp_sum;
        }

        return S;
    }
};
