#include <iostream>
#include "matrix/matrix.cpp"
#include "oneLayer/oneLayer.cpp"

int main(){
    //Creating our exponential matrix
    Matrix A(2,2);
    A.fillRandom(1,1);
    std::cout << "A" << std::endl;
    A.printMatrix();

    Matrix B(2,2);
    B.fillRandom(1,1);
    std::cout << "B" << std::endl;
    B.printMatrix();

    Matrix C(2,2);
    C = A+B;
    std::cout << "C" << std::endl;
    C.printMatrix();

    return(0);
}
