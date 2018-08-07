#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

class Matrix{
public:
    int rows;
	int cols;
	float* data;

    Matrix();
	Matrix(int n, int m);
	Matrix(int n);
	~Matrix();

    Matrix& operator=(const Matrix& secondMatrix);
    bool operator==(const Matrix& secondMatrix) const;
    Matrix operator+(const Matrix& secondMatrix) const;
    Matrix operator+(const RowVector& rowVector) const;
    Matrix operator-(const Matrix& secondMatrix) const;
    Matrix operator*(const Matrix& secondMatrix) const;
    Matrix operator*(const float& scalar) const;
    Matrix operator/(const float& scalar) const;
    Matrix& operator/=(const ColVector& colVector);

    void initializeZero();
	void fillMatrix(const float* values);
    void fillRandom(const int seed_num, const float dev);

    ColVector sumRows();
    void exponential();
    void printMatrix();
};
#endif
