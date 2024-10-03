#ifndef MPI_MODE_H
#define MPI_MODE_H

#include <mpi.h>
#include <vector>
#include <string>
#include <list>
using namespace std;


class SimplexMPI {
public:
    SimplexMPI(vector<vector<float>> matrix, vector<float> b, vector<float> c);
    vector<float> CalculateSimplex();
    void printResults(const std::string& mode);

private:
    vector<vector<float>> A;
    vector<float> B;
    vector<float> C;
    int rows, cols;
    float maximum;
    int rank, size;

    bool checkOptimality();
    int findPivotColumn();
    int findPivotRow(int pivotColumn);
    list<float> findFactor(int pivotRow, int pivotColumn);
    void doPivotting(int pivotRow, int pivotColumn);
};
#endif // MPI_MODE_H
