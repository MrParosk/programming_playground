#include "src/matrix.hpp"
#include "src/utils.hpp"
#include "src/one_layer_nn.hpp"
#include "src/ops.hpp"
#include <math.h>
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

void test_different_types()
{
    Matrix<int> A(2, 2);
    A.fill_vector(std::vector<int>({1, 2, 3, 4}));

    Matrix<int> B(2, 2);
    B.fill_vector(std::vector<int>({5, 6, 7, 8}));

    Matrix<int> C(2, 2);
    C.fill_vector(std::vector<int>({6, 8, 10, 12}));
    assert(A + B == C);
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

void test_multiply_scalar()
{
    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 2.0, 3.0, 4.0}));

    Matrix<float> B(2, 2);
    B.fill_vector(std::vector<float>({2.0, 4.0, 6.0, 8.0}));

    auto C = A * 2.0;
    assert(B == C);
}

void test_divide_operators()
{
    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({2.0, 4.0, 6.0, 8.0}));

    Matrix<float> B(2, 2);
    B.fill_vector(std::vector<float>({2.0, 2.0, 2.0, 2.0}));

    Matrix<float> C(2, 2);
    C.fill_vector(std::vector<float>({1.0, 2.0, 3.0, 4.0}));
    assert(A / B == C);
}

void test_equals()
{
    Matrix<float> A(2, 2);

    Matrix<float> B(2, 3);
    B.fill_vector(std::vector<float>({5.0, 6.0, 7.0, 8.0, 9.0, 10.0}));

    A = B;
    assert(A == B);
}

void test_transpose()
{
    Matrix<float> A(2, 1);
    A.fill_vector(std::vector<float>({1.0, 2.0}));

    Matrix<float> B(1, 2);
    B.fill_vector(std::vector<float>({1.0, 2.0}));

    auto C = A.transpose();
    assert(B == C);
}

void test_sum()
{
    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 2.0, 3.0, 4.0}));
    auto val = A.sum();

    assert(val == 10.0);
}

void test_sum_over_cols()
{
    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 2.0, 3.0, 4.0}));
    auto B = A.sum_over_cols();

    Matrix<float> C(2, 1);
    C.fill_vector(std::vector<float>({4.0, 6.0}));
    assert(B == C);
}

void test_exp()
{
    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 2.0, 3.0, 4.0}));

    Matrix<float> B(2, 2);
    B.fill_vector(std::vector<float>({exp(1.0f), exp(2.0f), exp(3.0f), exp(4.0f)}));

    assert(exponential(A) == B);
}

void test_softmax()
{
    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 9.0, 3.0, 4.0}));
    auto B = softmax(A);

    Matrix<float> C(2, 2);
    C.fill_vector(std::vector<float>({exp(1.0f) / (exp(1.0f) + exp(3.0f)),
                                      exp(9.0f) / (exp(9.0f) + exp(4.0f)),
                                      exp(3.0f) / (exp(1.0f) + exp(3.0f)),
                                      exp(4.0f) / (exp(9.0f) + exp(4.0f))}));
    assert(B == C);
}

void test_cross_entropy()
{
    const uint32_t num_samples = 1000;
    const uint32_t num_classes = 3;

    Matrix<float> Y(num_samples, num_classes);
    fill_random(Y, 0, 1.0);

    Matrix<float> Y_hat(num_samples, num_classes);
    fill_random(Y, 0, 1.0);

    CrossEntropy<float> ce;
    auto loss = ce.forward(Y_hat, Y);

    auto backwards = ce.backward(Y_hat, Y);
}

void test_one_layer()
{
    const uint32_t num_samples = 1000;
    const uint32_t num_features = 50;
    const uint32_t num_units = 100;
    const uint32_t num_classes = 3;

    OneLayer model(num_features, num_units, num_classes, 0, 1.0);

    Matrix<float> X(num_samples, num_features);
    fill_random(X, 0, 1.0);

    Matrix<float> Y(num_samples, num_classes);
    fill_random(Y, 0, 1.0);

    auto P = model.forward(X);
}

void run_tests()
{
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
    std::cout << "all tests passed" << std::endl;
}

int main()
{
    run_tests();
    return 0;
}
