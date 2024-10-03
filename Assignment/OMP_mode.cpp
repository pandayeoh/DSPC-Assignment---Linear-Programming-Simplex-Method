#include "OMP_mode.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <omp.h>  // OpenMP header
#include "Common.h"  // For the global isPrint flag
#include <limits>

using namespace std;

// Constructor: Initialize the SimplexOMP object with matrix A, vector B, and vector C
SimplexOMP::SimplexOMP(vector<vector<float>> matrix, vector<float> b, vector<float> c) {
    A = matrix;
    B = b;
    C = c;
    rows = matrix.size();
    cols = matrix[0].size();
    maximum = 0;
    omp_set_num_threads(4);
}

// Check if the current solution is optimal
bool SimplexOMP::checkOptimality() {
    for (int i = 0; i < cols; i++) {
        if (C[i] < 0) {
            return false; // If any of the coefficients in C are negative, it's not optimal
        }
    }
    return true;
}

// Find the pivot column with the smallest coefficient in the objective function (C vector)
int SimplexOMP::findPivotColumn() {
    int pivotColumn = 0;
    float minValue = C[0];

    // Do not parallelize as we're accessing both minValue and pivotColumn
    for (int i = 1; i < cols; i++) {
        if (C[i] < minValue) {
            minValue = C[i];
            pivotColumn = i;
        }
    }
    return pivotColumn;
}

// Find the pivot row with the smallest ratio of B[i] / A[i][pivotColumn]
int SimplexOMP::findPivotRow(int pivotColumn) {
    vector<float> ratios(rows, 0);
    int pivotRow = 0;
    float minRatio = std::numeric_limits<float>::max();

#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (A[i][pivotColumn] > 0) {
            ratios[i] = B[i] / A[i][pivotColumn];
        }
        else {
            ratios[i] = std::numeric_limits<float>::max(); // Avoid division by zero or negative pivot
        }
    }

    // Find the row with the minimum ratio
    for (int i = 0; i < rows; i++) {
        if (ratios[i] < minRatio) {
            minRatio = ratios[i];
            pivotRow = i;
        }
    }

    return pivotRow;
}

// Perform the pivot operation to update the tableau
void SimplexOMP::doPivotting(int pivotRow, int pivotColumn) {
    float pivotValue = A[pivotRow][pivotColumn];

    // Normalize the pivot row
#pragma omp parallel for
    for (int i = 0; i < cols; i++) {
        A[pivotRow][i] /= pivotValue;
    }
    B[pivotRow] /= pivotValue;

    // Update the other rows
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (i != pivotRow) {
            float factor = A[i][pivotColumn];
            for (int j = 0; j < cols; j++) {
                A[i][j] -= factor * A[pivotRow][j];
            }
            B[i] -= factor * B[pivotRow];
        }
    }

    // Update the objective function (C vector)
    float factor = C[pivotColumn];
    for (int i = 0; i < cols; i++) {
        C[i] -= factor * A[pivotRow][i];
    }
    maximum += factor * B[pivotRow];
}

// Calculate the Simplex solution using OpenMP parallelism
vector<float> SimplexOMP::CalculateSimplex() {
    while (!checkOptimality()) {
        int pivotColumn = findPivotColumn();
        int pivotRow = findPivotRow(pivotColumn);
        doPivotting(pivotRow, pivotColumn);
    }

    printResults("maximization");

    return B; // Return the solution (values of the variables)
}

// Print the results
void SimplexOMP::printResults(const string& mode) {
    if (isPrint) {
        cout << fixed << setprecision(2);
        cout << YELLOW << "Solution for the variables: " << RESET << endl;
        for (int i = 0; i < A.size(); i++)
        { // every basic column has the values, get it form B array
            int count0 = 0;
            int index = 0;
            for (int j = 0; j < rows; j++)
            {
                if (A[j][i] == 0.0)
                {
                    count0 += 1;
                }
                else if (A[j][i] == 1)
                {
                    index = j;
                }
            }

            if (count0 == rows - 1)
            {
                cout << GREEN << "Variable " << i + 1 << ": " << BOLD << B[index] << RESET << endl;
            }
            else
            {
                cout << GREEN << "Variable " << i + 1 << ": " << BOLD << 0 << RESET << endl;
            }
        }
    }
    cout << YELLOW << "The " << mode << " value of the objective function is: " << BOLD << maximum << RESET << endl;
}
