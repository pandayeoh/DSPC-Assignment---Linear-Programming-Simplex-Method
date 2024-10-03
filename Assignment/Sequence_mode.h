#ifndef SEQUENCE_MODE_H
#define SEQUENCE_MODE_H

#include <vector>
#include <string>
#include <list>

class Simplex {
public:
    Simplex(std::vector<std::vector<float>> matrix, std::vector<float> b, std::vector<float> c);
    std::vector<float> CalculateSimplex();  // Main function to calculate Simplex
    void printResults(const std::string& mode);

private:
    std::vector<std::vector<float>> A;  // Coefficient matrix
    std::vector<float> B;  // Right-hand side of the constraints
    std::vector<float> C;  // Coefficients of the objective function
    int rows, cols;  // Number of rows and columns
    float maximum;  // Objective function value (optimized result)

    bool checkOptimality();  // Check if the current solution is optimal
    void doPivotting(int pivotRow, int pivotColumn);  // Perform pivoting operation
    int findPivotColumn();  // Find the column to pivot
    int findPivotRow(int pivotColumn);  // Find the row to pivot
    std::list<float> findFactor(int pivotRow, int pivotColumn);
};

#endif  // SEQUENCE_MODE_H
