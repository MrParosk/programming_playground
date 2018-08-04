#ifndef SERIAL_GRADIENT_DESCENT_H
#define SERIAL_GRADIENT_DESCENT_H
#include "matrix.h"

void serial_comp_step_one(matrix* X, matrix* theta, matrix* y, matrix* placeholder);

void serial_comp_step_two(matrix* X_transpose, matrix* placeholder, matrix* gradient);

void serial_update_weights(matrix* theta, matrix* gradient, const float learning_rate, const float lambda);

void serial_gradient_descent(matrix* X, matrix* y, matrix* theta, const unsigned int num_iter, const float learning_rate, const float lambda);
#endif