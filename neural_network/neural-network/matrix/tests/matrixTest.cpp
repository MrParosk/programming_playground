#define CATCH_CONFIG_MAIN
#include "../../catch.hpp"
#include "../matrix.cpp"

TEST_CASE( "Testing + & - operations" ) {

    //The two matrices that will be used for operations
    Matrix A(2,2);
    float Aval [4] = {1.0, 2.0, 3.0, 4.0};
    A.fillMatrix(Aval);


    Matrix B(2,2);
    float Bval [4] = {5.0, 6.0, 7.0, 8.0};
    B.fillMatrix(Bval);

    //+ operator
    Matrix C(2,2);
    float Cval [4] = {6.0, 8.0, 10.0, 12.0};
    C.fillMatrix(Cval);
    REQUIRE(A+B == C);

    //- operator
    Matrix D(2,2);
    float Dval [4] = {-4.0, -4.0, -4.0, -4.0};
    D.fillMatrix(Dval);
    REQUIRE(A-B == D);

}

TEST_CASE( "Testing * & / operations for matrices" ) {

    //The two matrices that will be used for operations
    Matrix A(2,2);
    float Aval [4] = {1.0, 2.0, 3.0, 4.0};
    A.fillMatrix(Aval);

    Matrix B(2,3);
    float Bval [6] = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    B.fillMatrix(Bval);

    //Matrix multiplication
    Matrix C(2,3);
    float Cval [6] = {21.0, 24.0, 27.0, 47.0, 54.0, 61.0};
    C.fillMatrix(Cval);

    REQUIRE(A*B == C);

    //Scalar multiplication
    Matrix D(2,2);
    float Dval [4] = {2.0, 4.0, 6.0, 8.0};
    D.fillMatrix(Dval);

    REQUIRE(A*2.0 == D);

    //Scalar division
    Matrix E(2,2);
    float Eval [4] = {0.5, 1.0, 1.5, 2.0};
    E.fillMatrix(Eval);

    REQUIRE(A/2.0 == E);
}

TEST_CASE( "Testing exp and sum for matrices" ) {

    //The two matrices that will be used for operations
    Matrix A(2,2);
    float Aval [4] = {1.0, 2.0, 3.0, 4.0};
    A.fillMatrix(Aval);
    A.exponential();

    //Testing exp
    Matrix B(2,2);
    float Bval [4] = {exp(1.0), exp(2.0), exp(3.0), exp(4.0)};
    B.fillMatrix(Bval);

    REQUIRE(A == B);

    //Testing sumVectors
    Matrix C(2,2);
    float Cval [4] = {1.0, 2.0, 3.0, 4.0};
    C.fillMatrix(Cval);

    ColVector D(2);
    float Dval [2] = {3.0, 7.0};
    D.fillVector(Dval);

    REQUIRE(D == C.sumRows());

    //Testing division with col vector
    Matrix E(2,2);
    float Eval [4] = {1.0/3.0, 2.0/3.0, 3.0/7.0, 4.0/7.0};
    E.fillMatrix(Eval);

    C /= C.sumRows();
    REQUIRE(E == C);

}
