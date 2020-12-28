#include "src/matrix.hpp"
#include "src/utils.hpp"
#include <cassert>

void test_plus_minus_operators()
{

    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 2.0, 3.0, 4.0}));

    Matrix<float> B(2, 2);
    B.fill_vector(std::vector<float>({5.0, 6.0, 7.0, 8.0}));

    Matrix<float> C(2, 2);
    C.fill_vector(std::vector<float>({6.0, 8.0, 10.0, 12.0}));
    assert(A + B == C);

    Matrix<float> D(2, 2);
    D.fill_vector(std::vector<float>({-4.0, -4.0, -4.0, -4.0}));
    assert(A - B == D);
}

void test_multiply_operators()
{

    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 2.0, 3.0, 4.0}));

    Matrix<float> B(2, 3);
    B.fill_vector(std::vector<float>({5.0, 6.0, 7.0, 8.0, 9.0, 10.0}));

    Matrix<float> C(2, 3);
    C.fill_vector(std::vector<float>({23.0, 34.0, 31.0, 46.0, 39.0, 58.0}));
    assert(A * B == C);
}

void test_equals()
{
    Matrix<float> A(2, 2);

    Matrix<float> B(2, 3);
    B.fill_vector(std::vector<float>({5.0, 6.0, 7.0, 8.0, 9.0, 10.0}));

    A = B;
    assert(A == B);
}

void test_utils()
{
    Matrix<float> A(2, 2);
    fill_random(A, 0, 1.0);
    exponential(A);

    auto B = A.sum_over_cols();

    auto C = softmax(A);

    A.print_matrix();
    B.print_matrix();
    C.print_matrix();
}

int main()
{
    test_plus_minus_operators();
    test_multiply_operators();
    test_equals();
    test_utils();
    return 0;
}
