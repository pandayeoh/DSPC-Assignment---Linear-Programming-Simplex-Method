#ifndef OMP_MODE_H
#define OMP_MODE_H

#include <vector>
#include <string>

class SimplexOMP {
public:
    SimplexOMP(std::vector<std::vector<float>> matrix, std::vector<float> b, std::vector<float> c);
    std::vector<float> CalculateSimplex();  // Main function to calculate Simplex in parallel
    void printResults(const std::string& mode);

private:
    std::vector<std::vector<float>> A;  // Coefficient matrix
    std::vector<float> B;  // Right-hand side of the constraints
    std::vector<float> C;  // Coefficients of the objective function
    int rows, cols;  // Number of rows and columns
    float maximum;  // Objective function value (optimized result)

    bool checkOptimality();  // Check if the current solution is optimal (parallelized)
    void doPivotting(int pivotRow, int pivotColumn);  // Perform pivoting operation (parallelized)
    int findPivotColumn();  // Find the column to pivot (parallelized)
    int findPivotRow(int pivotColumn);  // Find the row to pivot (parallelized)
};

#endif  // OMP_MODE_H
