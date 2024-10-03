#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <list>
#include <limits>
#include "Common.h"  // For the global isPrint flag
#include "MPI_mode.h"  // For the global isPrint flag

using namespace std;

SimplexMPI::SimplexMPI(vector<vector<float>> matrix, vector<float> b, vector<float> c) {
    A = matrix;
    B = b;
    C = c;
    rows = matrix.size();
    cols = matrix[0].size();
    maximum = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

bool SimplexMPI::checkOptimality() {
    for (int i = 0; i < cols; i++) {
        if (C[i] < 0) {
            return false;
        }
    }
    return true;
}

int SimplexMPI::findPivotColumn() {
    int pivotColumn = 0;
    float minValue = C[0];
    for (int i = 1; i < cols; i++) {
        if (C[i] < minValue) {
            minValue = C[i];
            pivotColumn = i;
        }
    }
    return pivotColumn;
}

int SimplexMPI::findPivotRow(int pivotColumn) {
    vector<float> ratios(rows, 0);

    for (int i = 0; i < rows; i++) {
        if (A[i][pivotColumn] > 0) {
            ratios[i] = B[i] / A[i][pivotColumn];
        }
        else
        {
            ratios[i] = std::numeric_limits<float>::max();
        }
    }
    return min_element(ratios.begin(), ratios.end()) - ratios.begin();
}

list<float> SimplexMPI::findFactor(int pivotRow, int pivotColumn) {
    list<float> factorList;
    for (int i = 0; i < rows; i++) {
        if (i != pivotRow) {
            float factor = A[i][pivotColumn];
            factorList.push_back(factor);
        }
    }
    return factorList;
}

void SimplexMPI::doPivotting(int pivotRow, int pivotColumn) {
    float pivotValue = A[pivotRow][pivotColumn];

    //new code
    int tempSize = A[pivotRow].size() / size;
    vector<float> sub_A(tempSize, 0);

    MPI_Scatter(A[pivotRow].data(), tempSize, MPI_FLOAT, sub_A.data(), tempSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Broadcast the pivotValue to all ranks
    for (int i = 0; i < sub_A.size(); i++) {
        sub_A[i] /= pivotValue;
    }

    MPI_Gather(sub_A.data(), tempSize, MPI_FLOAT, A[pivotRow].data(), tempSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(A[pivotRow].data(), cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        B[pivotRow] /= pivotValue;
    }

    MPI_Bcast(&B[pivotRow], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Parallel row updates (each rank will handle a subset of rows)
    for (int i = rank; i < rows; i += size) {
        if (i != pivotRow) {
            float factor = A[i][pivotColumn];
            for (int j = 0; j < cols; j++) {
                A[i][j] -= factor * A[pivotRow][j];
            }
            B[i] -= factor * B[pivotRow];
        }
    }

    // Gather the updated rows across all processes
    for (int i = 0; i < rows; i++) {
        MPI_Bcast(&A[i][0], cols, MPI_FLOAT, i % size, MPI_COMM_WORLD);
        MPI_Bcast(&B[i], 1, MPI_FLOAT, i % size, MPI_COMM_WORLD);
    }

    // Update the objective function on rank 0
    if (rank == 0) {
        float factor = C[pivotColumn];
        for (int i = 0; i < cols; i++) {
            C[i] -= factor * A[pivotRow][i];
        }
        maximum += factor * B[pivotRow];
    }

    // Broadcast the updated C vector and maximum value to all processes
    MPI_Bcast(&C[0], cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maximum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}



vector<float> SimplexMPI::CalculateSimplex() {
    while (!checkOptimality()) {
        int pivotColumn = findPivotColumn();
        int pivotRow = findPivotRow(pivotColumn);
        doPivotting(pivotRow, pivotColumn);
    }

    if(rank == 0)
        printResults("maximization");


    return B;
}

// Print the results
void SimplexMPI::printResults(const string& mode) {
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