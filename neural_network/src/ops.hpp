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
        throw "forward is not implemented yet";
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
        auto ce = Y.transpose() * Y_hat_temp;
        auto loss = ce * ((T)-1.0);
        return loss.sum();
    }

    Matrix<T> backward(Matrix<T> &Y_hat, Matrix<T> &Y)
    {
        // TODO: add this
        return Y_hat;
    }
};
