#include "src/matrix.hpp"
#include "src/utils.hpp"
#include "src/one_layer_nn.hpp"
#include "src/ops.hpp"
#include "src/tests/tests.hpp"

void run_training()
{
    const float learning_rate = 0.01f;
    const uint32_t num_samples_per_class = 200;
    const uint32_t num_features = 2;
    const uint32_t num_units = 10;
    const uint32_t num_classes = 2;

    std::vector<std::vector<float>> center_means({{1.0, 1.0}, {-1.0, -1.0}});

    Matrix<float> X, Y;
    std::tie(X, Y) = generate_data<float>(center_means, num_samples_per_class, num_features, num_classes);

    // One column for the bias term
    OneLayer<float> model(learning_rate, num_features + 1, num_units, num_classes);

    for (uint32_t i = 0; i < 100; i++)
    {
        model.step(X, Y);
    }

    auto Y_hat = model.forward(X);
    std::cout << "Model accuracy: " << accuracy(Y, Y_hat) << std::endl;
}

int main()
{
    run_tests();
    run_training();
    return 0;
}
