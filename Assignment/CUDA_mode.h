#ifndef CUDA_MODE_H
#define CUDA_MODE_H

#include <vector>
#include <string>
#include <list>
using namespace std;


class SimplexCUDA {
public:
    // Constructor
    SimplexCUDA(std::vector<std::vector<float>> matrix, std::vector<float> b, std::vector<float> c);
    std::vector<float> CalculateSimplex();
    void printResults(const std::string& mode);


private:
    // Member variables
    std::vector<float> A_flat;   // Flattened matrix A
    std::vector<vector<float>> A;   // Flattened matrix A
    std::vector<float> B;        // Vector B
    std::vector<float> C;        // Vector C (objective function coefficients)
    int rows;                    // Number of rows in A
    int cols;                    // Number of columns in A
    float maximum;               // Maximum value of the objective function

    // Device pointers for GPU memory
    float* d_A;          // Device pointer for matrix A
    float* d_B;          // Device pointer for vector B
    float* d_C;          // Device pointer for vector C
    float* d_maximum;    // Device pointer for maximum value
    bool* d_isOptimal;   // Device pointer for optimality flag
    int* d_pivotColumn;  // Device pointer for pivot column index
    int* d_pivotRow;     // Device pointer for pivot row index

};

#endif // CUDA_MODE_H