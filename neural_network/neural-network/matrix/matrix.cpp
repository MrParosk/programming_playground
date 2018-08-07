#include <random>
#include <iostream>
#include <memory>
#include <math.h>

#include "matrix.h"
#include "vector.cpp"

Matrix::Matrix(int n, int m){
    rows = n;
    cols = m;

    data = new float[rows*cols];
    Matrix::initializeZero();
}

Matrix::Matrix(int n){
    rows = n;
    cols = n;

    data = new float[rows*cols];
    Matrix::initializeZero();
}

Matrix::Matrix(){}

Matrix::~Matrix(){
    delete[] data;
    data = NULL;
    rows = 0;
    cols = 0;
}

Matrix& Matrix::operator=(const Matrix& secondMatrix){

    rows = secondMatrix.rows;
	cols = secondMatrix.cols;

    delete[] data;
	data = NULL;
	data = new float[rows*cols];

	for(int i=0; i<rows*cols; i++){
			data[i] = secondMatrix.data[i];
	}

	return *this;
}

bool Matrix::operator==(const Matrix& secondMatrix) const{
    if(rows != secondMatrix.rows || cols != secondMatrix.cols){
		return false;
	}

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			if(fabs(data[j+i*cols] - secondMatrix.data[j+i*secondMatrix.cols]) > 1e-6){
				return false;
			}
		}
	}

	return true;
}

Matrix Matrix::operator+(const Matrix& secondMatrix) const{
    if(rows != secondMatrix.rows || cols != secondMatrix.cols){
        throw std::runtime_error("Matrix dimensions does not match for addition of matrices");
    }

    Matrix returnMatrix(rows,cols);

    for(int i=0; i<rows*cols; ++i){
        returnMatrix.data[i] = data[i] + secondMatrix.data[i];
    }

    return returnMatrix;
}

Matrix Matrix::operator+(const RowVector& rowVector) const{
    if(rows != rowVector.rows){
        throw std::runtime_error("Matrix dimensions does not match for addition with row vector");
    }

    Matrix returnMatrix(rows,cols);

    for(int j=0;j<cols;++j){
        for(int i=0;i<rows;++i){
            returnMatrix.data[i+j*cols] = data[i+j*cols] + rowVector.data[i];
        }
    }

    return returnMatrix;
}

Matrix Matrix::operator-(const Matrix& secondMatrix) const{
    if(rows != secondMatrix.rows || cols != secondMatrix.cols){
        throw std::runtime_error("Matrix dimensions does not match for subtraction of matrices");
    }

    Matrix returnMatrix(rows,cols);

    for(int i=0; i<rows*cols; ++i){
        returnMatrix.data[i] = data[i] - secondMatrix.data[i];
    }

    return returnMatrix;
}

Matrix Matrix::operator*(const Matrix& secondMatrix) const{
    if(cols != secondMatrix.rows){
        throw std::runtime_error("Matrix dimensions does not match for multiplication of matrices");
    }

    Matrix returnMatrix(rows,secondMatrix.cols);

    for(int j=0;j<secondMatrix.cols;++j){
        for(int i=0;i<rows;++i){
            float tempSum = 0;
            for(int k=0;k<cols;k++){
                tempSum += data[k+i*cols]*secondMatrix.data[j+k*secondMatrix.cols];
            }
            returnMatrix.data[i*secondMatrix.cols + j] = tempSum;
        }
    }

    return returnMatrix;
}

Matrix Matrix::operator*(const float& scalar) const{
    Matrix returnMatrix(rows,cols);

    for(int i=0;i<rows*cols;++i){
        returnMatrix.data[i] = scalar*data[i];
    }

    return returnMatrix;
}

Matrix Matrix::operator/(const float& scalar) const{
    Matrix returnMatrix(rows,cols);

    for(int i=0;i<rows*cols;++i){
        returnMatrix.data[i] = data[i]/scalar;
    }

    return returnMatrix;
}

Matrix& Matrix::operator/=(const ColVector& colVector){
    if(cols != colVector.cols){
        throw std::runtime_error("Matrix dimensions does not match for division with column vector");
    }

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            data[i+j*rows] /= colVector.data[j];
        }
    }

    return *this;
}

void Matrix::initializeZero(){
    for(int i=0; i<rows*cols; ++i){
        data[i] = 0;
    }
}

void Matrix::fillMatrix(const float* values){
    for(int i=0; i<rows*cols; i++){
        data[i] = values[i];
    }
}

void Matrix::fillRandom(const int seed_num, const float dev){

    std::default_random_engine seed(seed_num); //seed
    std::normal_distribution<float> generator(0, 1); // Sample from zero mean and unit variance

    for(int i=0; i<rows*cols; ++i){
        data[i] = dev*generator(seed);
    }
}

ColVector Matrix::sumRows(){
    ColVector sumVector(cols);

    for(int j=0;j<cols;j++){
        float tempSum = 0;

        for(int i=0;i<rows;i++){
            tempSum += data[i+j*cols];
        }

        sumVector.data[j] = tempSum;
    }

    return sumVector;
}

void Matrix::exponential(){
    for(int i=0;i<rows*cols;i++){
        data[i] = exp(data[i]);
    }
}

void Matrix::printMatrix(){
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            std::cout << data[i+j*rows] << " ";
        }
        std::cout << std::endl;
    }
}
