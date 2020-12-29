#pragma once
#include "matrix.hpp"

class Ops
{
public:
    Ops() {}
    virtual void forward()
    {
        throw "forward is not implemented yet";
    }
    virtual void backward()
    {
        throw "backward is not implemented yet";
    }
};

template <class T>
class CrossEntropy : Ops
{
public:
    CrossEntropy() {}

    T forward(Matrix<T> &Y_hat, Matrix<T> &Y)
    {
        auto Y_hat_temp = Y_hat.copy();
        log(Y_hat_temp);
        return ((Y.transpose() * Y_hat_temp) * ((T)-1.0)).sum();
    }

    Matrix<T> backward(Matrix<T> &Y_hat, Matrix<T> &Y)
    {
        return (Y / Y_hat - ((Y - 1) / (Y_hat - 1))) * -(T)1.0;
    }
};
