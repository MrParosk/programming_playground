#include "../matrix/matrix.h"
#include "oneLayer.h"

OneLayer::OneLayer(const Matrix& X_, const Matrix& Y_, const int units, const int seed, const float dev){
    X = X_;
    Y = Y_;

    W1 = Matrix(units, X.rows);
    W1.fillRandom(seed, dev);
    W2 = Matrix(Y.rows, units);
    W2.fillRandom(seed+1, dev);

    b1 = RowVector(units);
    b1.fillRandom(seed+2, dev);
    b2 = RowVector(Y.rows);
    b2.fillRandom(seed+3, dev);
}

Matrix OneLayer::softmax(Matrix s) {

    Matrix P(Y.rows, X.cols);

    ColVector sumClasses = s.sumRows();
    P.exponential();

    P /= sumClasses;
    return P;
}

Matrix OneLayer::Relu(Matrix h){
    for(int i=0;i<h.rows*h.cols;i++){
        if(h.data[i] < 0){
            h.data[i] = 0;
        }
    }

    return h;
}

Matrix OneLayer::forwardPass(){
    Matrix h;
    Matrix s;
    Matrix P;

    h = Relu(W1*X + b1);
    s = W2*h + b2;
    P = softmax(s);
    return P;
}
