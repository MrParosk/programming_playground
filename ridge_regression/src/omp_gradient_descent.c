#include <omp.h>
#include "matrix.h"

void omp_comp_step_one(matrix* X, matrix* theta, matrix* y, matrix* placeholder){
    /*
    Doing the first computation step, i.e. (X * theta - y) 
        - X matrix of size [num_samples, num_features]
        - theta of size [num_features, 1]
        - y of size [num_samples, 1]
        - placeholder of size [num_samples, 1]
    */

    // Only one parallell for-loop since X->rows >> X->cols
    #pragma omp parallel for
    for(unsigned int i = 0; i < X->rows; i++){
        float temp_sum = 0.0f;

        for(unsigned int j = 0; j < X->cols; j++){
            temp_sum += X->values[j + i * X->cols] * theta->values[j];
        }
        placeholder->values[i] = (temp_sum - y->values[i]);
    }
}

void omp_comp_step_two(matrix* X_transpose, matrix* placeholder, matrix* gradient){
    /*
        Doing the second computation step, i.e. X.T * placeholder / num_samples
            - X.T matrix of size [num_features, num_samples]
            - placeholder is the output from omp_comp_step_one
            - gradient of size [num_features, 1]
    */

    for(unsigned int i = 0; i < X_transpose->rows; i++){
        float temp_sum = 0.0f;

        // Only one parallell for-loop since X_transpose->cols >> X_transpose->rows
        #pragma omp parallel for reduction(+:temp_sum)
        for(unsigned int j = 0; j < X_transpose->cols; j++){
            temp_sum += X_transpose->values[j + i * X_transpose->cols] * placeholder->values[j];
        }
        gradient->values[i] = temp_sum/(X_transpose->cols);
    }
}

void omp_update_weights(matrix* theta, matrix* gradient, const float learning_rate, const float lambda){
    for(unsigned int i = 0; i < theta->rows; i++){
        theta->values[i] = theta->values[i] - learning_rate * (gradient->values[i] + (lambda * theta->values[i])/(2.0f));
    }
}

void omp_gradient_descent(matrix* X, matrix* y, matrix* theta, const unsigned int num_iter, const float learning_rate, const float lambda){
    const unsigned int num_samples = X->rows;
    const unsigned int num_features = X->cols;

    matrix* X_transpose = create_matrix(num_samples, num_features);
    equal_matrix(X_transpose, X);
    
    transpose_matrix(X_transpose);

    matrix* placeholder = create_matrix(num_samples, 1);
    matrix* gradient = create_matrix(num_features, 1);

    for(unsigned int i = 0; i < num_iter; i++){
        omp_comp_step_one(X, theta, y, placeholder);
        omp_comp_step_two(X_transpose, placeholder, gradient);
        omp_update_weights(theta, gradient, learning_rate, lambda);
    }

    free_matrix(X_transpose);
    free_matrix(placeholder);
    free_matrix(gradient);
}
