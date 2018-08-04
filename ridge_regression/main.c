#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "src/matrix.h"
#include "src/utils.h"
#include "src/serial_gradient_descent.h"
#include "src/omp_gradient_descent.h"
#define NUM_THREADS 4

int main(){
    const unsigned int num_samples = 1000000;
    const unsigned int num_features = 5;

    const unsigned int num_iter = 200;
    const float learning_rate = 0.05f;
    const float lambda = 1e-3f;

    omp_set_num_threads(NUM_THREADS);

    float* X_values = load_data("data/X.txt", num_samples, num_features);
    matrix* X = create_matrix(num_samples, num_features);
    fill_matrix_values(X, X_values);
    free(X_values);

    float* y_values = load_data("data/y.txt", num_samples, 1);
    matrix* y = create_matrix(num_samples, 1);
    fill_matrix_values(y, y_values);
    free(y_values);

    printf("------------------Serial implemenation------------------ \n");
    matrix* serial_theta = create_matrix(num_features, 1);
    fill_matrix_random(serial_theta);

    // True theta = [1.5, 2.2, 3.5, -2.3, -0.5]
    double serial_begin = omp_get_wtime();
    serial_gradient_descent(X, y, serial_theta, num_iter, learning_rate, lambda);
    double serial_end = omp_get_wtime();  
    printf("It took %f s \n", serial_end - serial_begin);

    printf("Theta values: \n");
    print_matrix(serial_theta);
    float serial_rmse_value = rmse(X, serial_theta, y);
    printf("\n");
    printf("RMSE value: %f \n", serial_rmse_value);
    free_matrix(serial_theta);

    printf("------------------Parallel implemenation------------------ \n");
    matrix* omp_theta = create_matrix(num_features, 1);
    fill_matrix_random(omp_theta);

    // True theta = [1.5, 2.2, 3.5, -2.3, -0.5]
    double omp_begin = omp_get_wtime();
    omp_gradient_descent(X, y, omp_theta, num_iter, learning_rate, lambda);
    double omp_end = omp_get_wtime();  
    printf("It took %f s \n", omp_end - omp_begin);

    printf("Theta values: \n");
    print_matrix(omp_theta);
    float omp_rmse_value = rmse(X, omp_theta, y);
    printf("\n");
    printf("RMSE value: %f \n", omp_rmse_value);
    free_matrix(omp_theta);

    free_matrix(X);
    free_matrix(y);
    
    return 0;
}
