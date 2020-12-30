#include "test_matrix.hpp"
#include "test_ops.hpp"

void run_tests()
{
    // Doing simple asserts instead of including a test-library to keep the project self-contained.
    test_plus_minus_operators();
    test_different_types();
    test_multiply_operators();
    test_multiply_scalar();
    test_divide_operators();
    test_equals();
    test_transpose();
    test_sum();
    test_sum_over_cols();
    test_exp();
    test_softmax();
    test_cross_entropy();
    test_one_layer();
}
