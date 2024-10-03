#include "CUDA_mode.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Common.h"  // For the global isPrint flag

using namespace std;

// Constructor: Initialize the SimplexCUDA object with matrix A, vector B, and vector C
SimplexCUDA::SimplexCUDA(vector<vector<float>> matrix, vector<float> b, vector<float> c) {
    // Flatten matrix A into a single 1D array
    A_flat = vector<float>(matrix.size() * matrix[0].size());
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[0].size(); j++) {
            A_flat[i * matrix[0].size() + j] = matrix[i][j];  // Flatten A into 1D array
        }
    }

    A = matrix;
    B = b;
    C = c;
    rows = matrix.size();
    cols = matrix[0].size();
    maximum = 0;

    // Allocate device memory
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, rows * sizeof(float));
    cudaMalloc(&d_C, cols * sizeof(float));
    cudaMalloc(&d_maximum, sizeof(float));
    cudaMalloc(&d_isOptimal, sizeof(bool));
    cudaMalloc(&d_pivotColumn, sizeof(int));
    cudaMalloc(&d_pivotRow, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_A, A_flat.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    float h_maximum = 0;
    cudaMemcpy(d_maximum, &h_maximum, sizeof(float), cudaMemcpyHostToDevice);
}

// Device function to find the pivot column
__device__ void findPivotColumnDevice(float* C, int cols, bool* isOptimal, int* pivotColumn) {
    int tid = threadIdx.x;
    __shared__ float minValue;
    __shared__ int minIndex;

    if (tid == 0) {
        minValue = C[0];
        minIndex = 0;
        for (int i = 1; i < cols; i++) {
            if (C[i] < minValue) {
                minValue = C[i];
                minIndex = i;
            }
        }
        *isOptimal = (minValue >= 0);
        *pivotColumn = minIndex;
    }
    __syncthreads();
}

// Device function to find the pivot row
__device__ void findPivotRowDevice(float* A, float* B, int rows, int cols, int pivotColumn, int* pivotRow) {
    int tid = threadIdx.x;
    __shared__ float minRatio;
    __shared__ int minIndex;

    if (tid == 0) {
        minRatio = FLT_MAX;
        minIndex = -1;
        for (int i = 0; i < rows; i++) {
            float a = A[i * cols + pivotColumn];
            if (a > 0) {
                float ratio = B[i] / a;
                if (ratio < minRatio) {
                    minRatio = ratio;
                    minIndex = i;
                }
            }
        }
        *pivotRow = minIndex;
    }
    __syncthreads();
}

// Kernel to perform one iteration of the simplex method
__global__ void simplexIteration(float* A, float* B, float* C, int rows, int cols, float* maximum, bool* isOptimal, int* pivotColumn, int* pivotRow) {
    if (threadIdx.x == 0) {
        // Initialize isOptimal flag
        *isOptimal = true;
    }
    __syncthreads();

    // Find pivot column
    findPivotColumnDevice(C, cols, isOptimal, pivotColumn);
    __syncthreads();

    if (*isOptimal) {
        return;  // Optimal solution found
    }

    // Find pivot row
    findPivotRowDevice(A, B, rows, cols, *pivotColumn, pivotRow);
    __syncthreads();

    // Get pivot value
    float pivotValue = A[*pivotRow * cols + *pivotColumn];

    // Normalize pivot row
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        A[*pivotRow * cols + j] /= pivotValue;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        B[*pivotRow] /= pivotValue;
    }
    __syncthreads();

    // Update other rows
    for (int i = 0; i < rows; i++) {
        if (i != *pivotRow) {
            float factor = A[i * cols + *pivotColumn];
            for (int j = threadIdx.x; j < cols; j += blockDim.x) {
                A[i * cols + j] -= factor * A[*pivotRow * cols + j];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                B[i] -= factor * B[*pivotRow];
            }
            __syncthreads();
        }
    }

    // Update objective function
    float factor = C[*pivotColumn];
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        C[j] -= factor * A[*pivotRow * cols + j];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *maximum += factor * B[*pivotRow];
    }
    __syncthreads();
}

// Calculate the Simplex solution
vector<float> SimplexCUDA::CalculateSimplex() {
    bool h_isOptimal = false;

    // Loop until optimality is reached
    while (!h_isOptimal) {
        // Launch the simplex iteration kernel
        simplexIteration << <1, 1024 >> > (d_A, d_B, d_C, rows, cols, d_maximum, d_isOptimal, d_pivotColumn, d_pivotRow);
        cudaDeviceSynchronize();

        // Copy optimality flag back to host
        cudaMemcpy(&h_isOptimal, d_isOptimal, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // Copy results back to host
    cudaMemcpy(A_flat.data(), d_A, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B.data(), d_B, rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C.data(), d_C, cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maximum, d_maximum, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_maximum);
    cudaFree(d_isOptimal);
    cudaFree(d_pivotColumn);
    cudaFree(d_pivotRow);

    printResults("maximization");

    return B; // Return the solution (values of the variables)
}

// Print the results
void SimplexCUDA::printResults(const string& mode) {
    if (isPrint) {
        cout << fixed << setprecision(2);
        cout << YELLOW << "Solution for the variables: " << RESET << endl;
        int numRows = A.size();
        int numCols = A[0].size();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                A[i][j] = A_flat[i * numCols + j];  // Convert 1D array back to 2D
            }
        }
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
