#include <random>
#include <iostream>
#include "vector.h"

RowVector::RowVector(){}

RowVector::RowVector(int n){
    rows = n;

    data = new float[n];
    RowVector::initializeZero();
}

RowVector& RowVector::operator=(const RowVector& secondVector){

    rows = secondVector.rows;

    delete[] data;
	data = NULL;
	data = new float[rows];

	for(int i=0; i<rows; i++){
			data[i] = secondVector.data[i];
	}

	return *this;
}

bool RowVector::operator==(const RowVector& secondVector) const{
    if(rows != secondVector.rows){
		return false;
	}

	for(int i=0; i<rows; i++){
			if(fabs(data[i] - secondVector.data[i]) > 1e-6){
				return false;
			}
    }

	return true;
}

void RowVector::initializeZero(){
    for(int i=0; i<rows; ++i){
        data[i] = 0;
    }
}

void RowVector::fillVector(const float* values){
    for(int i=0; i<rows; i++){
        data[i] = values[i];
    }
}

void RowVector::fillRandom(const int seed_num, const float dev){

    std::default_random_engine seed(seed_num); //seed
    std::normal_distribution<float> generator(0, 1); // Sample from zero mean and unit variance

    for(int i=0; i<rows; ++i){
        data[i] = dev*generator(seed);
    }
}

void RowVector::printVector(){
    for(int i=0;i<rows;++i){
            std::cout << data[i] << std::endl;
    }
}

ColVector::ColVector(){}

ColVector::ColVector(int m){
    cols = m;

    data = new float[m];
    ColVector::initializeZero();
}

ColVector& ColVector::operator=(const ColVector& secondVector){

    cols = secondVector.cols;

    delete[] data;
	data = NULL;
	data = new float[cols];

	for(int i=0; i<cols; i++){
			data[i] = secondVector.data[i];
	}

	return *this;
}

bool ColVector::operator==(const ColVector& secondVector) const{
    if(cols != secondVector.cols){
		return false;
	}

	for(int i=0; i<cols; i++){
			if(fabs(data[i] - secondVector.data[i]) > 1e-6){
				return false;
			}
    }

	return true;
}

void ColVector::initializeZero(){
    for(int i=0; i<cols; ++i){
        data[i] = 0;
    }
}

void ColVector::fillVector(const float* values){
    for(int i=0; i<cols; i++){
        data[i] = values[i];
    }
}

void ColVector::fillRandom(const int seed_num, const float dev){

    std::default_random_engine seed(seed_num); //seed
    std::normal_distribution<float> generator(0, 1); // Sample from zero mean and unit variance

    for(int i=0; i<cols; ++i){
        data[i] = dev*generator(seed);
    }
}

void ColVector::printVector(){
    for(int i=0;i<cols;++i){
            std::cout << data[i] << " ";
    }

    std::cout << std::endl;
}
