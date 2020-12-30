#include "../matrix.hpp"
#include "../utils.hpp"
#include "../one_layer_nn.hpp"
#include "../ops.hpp"

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
    SoftmaxCrossEntropy<float> sm;
    Matrix<float> A(2, 2);
    A.fill_vector(std::vector<float>({1.0, 9.0, 3.0, 4.0}));
    auto B = sm.forward_softmax(A);

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
    fill_random(Y, 0, 1.0f);

    Matrix<float> Y_hat(num_samples, num_classes);
    fill_random(Y, 0, 1.0f);

    SoftmaxCrossEntropy<float> ce;

    Y = ce.forward_softmax(Y);
    ce.forward_cross_entropy(Y_hat, Y);
    ce.backward(Y_hat, Y);
}

void test_one_layer()
{
    const float learning_rate = 0.1f;
    const uint32_t num_samples = 1000;
    const uint32_t num_features = 50;
    const uint32_t num_units = 100;
    const uint32_t num_classes = 3;

    OneLayer<float> model(learning_rate, num_features, num_units, num_classes);

    Matrix<float> X(num_samples, num_features);
    fill_random(X, 0, 0.1f);

    Matrix<float> Y(num_samples, num_classes);
    fill_random(Y, 0, 0.1f);
    SoftmaxCrossEntropy<float> ce;
    Y = ce.forward_softmax(Y);

    model.step(X, Y);
}
