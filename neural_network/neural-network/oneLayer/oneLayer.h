#ifndef ONELAYER_H
#define ONELAYER_H

class OneLayer{
public:
    Matrix X;
    Matrix Y;

    Matrix W1;
    Matrix W2;

    RowVector b1;
    RowVector b2;

    OneLayer(const Matrix& X_, const Matrix& Y_, int units, const int seed, const float dev);
    Matrix softmax(Matrix s);
    Matrix Relu(Matrix h);
    Matrix forwardPass();
};
#endif
