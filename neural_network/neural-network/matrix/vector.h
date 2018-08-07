#ifndef VECTOR_H
#define VECTOR_H

//Vector on the format [n,1]
class RowVector {
    public:
        int rows;
        float* data;

    RowVector();
    RowVector(int n);
    RowVector& operator=(const RowVector& secondVector);
    bool operator==(const RowVector& secondVector) const;
    void initializeZero();
    void fillVector(const float* values);
    void fillRandom(const int seed_num, const float dev);
    void printVector();

};

//Vector on the format [1,m]
class ColVector {
    public:
        int cols;
        float* data;

    ColVector();
    ColVector(int n);
    ColVector& operator=(const ColVector& secondVector);
    bool operator==(const ColVector& secondVector) const;
    void initializeZero();
    void fillVector(const float* values);
    void fillRandom(const int seed_num, const float dev);
    void printVector();

};

#endif
