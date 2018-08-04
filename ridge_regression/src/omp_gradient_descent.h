#ifndef OMP_GRADIENT_DESCENT_H
#define OMP_GRADIENT_DESCENT_H
#include "matrix.h"

void omp_comp_step_one(matrix* X, matrix* theta, matrix* y, matrix* placeholder);

void omp_comp_step_two(matrix* X_transpose, matrix* placeholder, matrix* gradient);

void omp_update_weights(matrix* theta, matrix* gradient, const float learning_rate, const float lambda);

void omp_gradient_descent(matrix* X, matrix* y, matrix* theta, const unsigned int num_iter, const float learning_rate, const float lambda);

#endif