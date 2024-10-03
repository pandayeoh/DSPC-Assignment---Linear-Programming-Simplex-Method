#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <iomanip>  // For table formatting
#include "Sequence_mode.h"
#include "Common.h"
#ifdef _OPENMP
#include "OMP_mode.h"  // Include OpenMP header if available
#endif
#ifdef USE_CUDA
#include "CUDA_mode.h"  // Include CUDA header if available
#endif
#ifdef USE_MPI
#include "MPI_mode.h"   // Include MPI header if available
#endif

using namespace std;
using namespace chrono;

bool isPrint = false;

const string RESET = "\033[0m";
const string RED = "\033[31m";
const string GREEN = "\033[32m";
const string YELLOW = "\033[33m";
const string BLUE = "\033[34m";
const string MAGENTA = "\033[35m";
const string CYAN = "\033[36m";
const string BOLD = "\033[1m";

// Function to read the CSV and parse A, B, and C vectors
void readCSV(const string& filename, vector<vector<float>>& A, vector<float>& B, vector<float>& C) {
    ifstream file(filename);
    string line;
    vector<string> allLines;

    // Read all lines from the CSV and store them in memory
    while (getline(file, line)) {
        allLines.push_back(line);
    }

    // The last line is the C vector (objective function)
    stringstream lastLine(allLines.back());
    allLines.pop_back();  // Remove the last line from allLines

    string item;
    vector<float> c_row;

    // Skip the first column (food item name) in the last row (C vector)
    getline(lastLine, item, ',');  // Ignore the first column

    // Read the remaining values as the C vector (objective function)
    while (getline(lastLine, item, ',')) {
        if (!item.empty()) {
            c_row.push_back(stof(item));  // Convert string to float
        }
    }

    // The C vector is now stored in c_row (excluding the B value, which is None in this case)
    C.assign(c_row.begin(), c_row.end() - 1);  // The last value in the row is ignored as it’s not part of C

    // Process the rest of the rows as part of A matrix and B vector
    for (int i = 1; i < allLines.size(); i++) {  // Start from 1 to skip the first row
        stringstream ss(allLines[i]);
        vector<float> row;

        // Skip the first column (food item name)
        getline(ss, item, ',');

        // Read the rest of the values (Protein, Fat, Cholesterol, Vitamin C, and B)
        while (getline(ss, item, ',')) {
            if (!item.empty()) {
                row.push_back(stof(item));  // Convert string to float
            }
        }

        // First columns go to A matrix, and the last column goes to B vector
        vector<float> A_row(row.begin(), row.end() - 1);  // All but the last column go to A
        A.push_back(A_row);
        B.push_back(row.back());  // The last column goes to B
    }

    int tempSize = C.size();
    for (int i = 0; i < tempSize; i++) {
        C.push_back(0);
    }

    for (int i = 0; i < A.size(); i++) {
        int tempSize = A[i].size();
        for (int j = 0; j < tempSize; j++) {
            if (i == j)
            {
                A[i].push_back(1);
            }
            else
            {
                A[i].push_back(0);
            }
        }
    }

    // Check the size of matrix A
    cout << GREEN << "Size of matrix A: " << BOLD << A.size() << " rows, " << (A.empty() ? 0 : A[0].size()) << " columns" << RESET << endl;

    // Check the size of vector B
    cout << GREEN << "Size of vector B: " << BOLD << B.size() << RESET << endl;

    // Check the size of vector C
    cout << GREEN << "Size of vector C: " << BOLD << C.size() << RESET << endl;

    cout << BOLD << GREEN << "CSV file read successfully." << RESET << endl;
}

string floatToString(float number) {
    std::ostringstream oss;
    // Set fixed-point notation, 2 decimal places
    oss << std::fixed << std::setprecision(2);
    // Set the fill character to '0' and the width to 6 (3 digits + decimal point + 2 decimals)
    oss << std::setw(6) << std::setfill('0') << number;
    return oss.str();
}

bool areVectorsEqual(const vector<float>& vec1, const vector<float>& vec2, float epsilon = 1e-5) {
    if (vec1.size() != vec2.size()) {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        if (fabs(vec1[i] - vec2[i]) > epsilon) {
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <CSV file> <mode (sequence/omp/cuda/mpi/all)>" << endl;
        //return 1;
    }

    string csv_file = "";

    if (argv[1] != "") {
        csv_file = "Dataset/" + string(argv[1]) + ".csv";
    }
    else {
        csv_file = "Dataset/test.csv";
    }

    string mode = (argv[2]!="")? argv[2]:"ALL";

    isPrint = string(argv[3]) == "YES";

    //string csv_file = "x64/Debug/Dataset/simplified_ingredients.csv";
    //string mode = "OMP";

    // Initialize data
    vector<vector<float>> A;  // Constraints coefficients
    vector<float> B;  // Right-hand side of constraints
    vector<float> C;  // Objective function coefficients
    int rank = -1;

    // Read the CSV file
    if (mode == "MPI" || mode == "ALL") {
        MPI_Init(&argc, &argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0) {
            cout << YELLOW << "\nReading CSV file" << BOLD << "..." << RESET << endl;
            readCSV(csv_file, A, B, C);
        }
        int rows = A.size();
        int cols = (rows > 0) ? A[0].size() : 0;
        
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize A, B, and C on non-root processes
        if (rank != 0) {
            A.resize(rows, vector<float>(cols));
            B.resize(rows);
            C.resize(cols);
        }

        // Broadcast matrix A
        for (int i = 0; i < rows; i++) {
            MPI_Bcast(A[i].data(), cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Broadcast vector B
        MPI_Bcast(B.data(), rows, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Broadcast vector C
        MPI_Bcast(C.data(), cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else {
        readCSV(csv_file, A, B, C);
    }

    // Variables to hold the run times
    double seq_time = 0, omp_time = 0, cuda_time = 0, mpi_time = 0;
    vector<float> seqR, ompR, cudaR, mpiR;
    auto start = high_resolution_clock::now();
    auto end = high_resolution_clock::now();

    // Sequence mode (always runs)
    if (mode == "SEQUENCE" || (mode == "ALL" && rank == 0)) {
        cout << YELLOW << "\nRunning Sequence mode" << BOLD << "..." << RESET << endl;
        Simplex simplex(A, B, C);
        start = high_resolution_clock::now();
        seqR = simplex.CalculateSimplex();
        end = high_resolution_clock::now();
        seq_time = duration_cast<milliseconds>(end - start).count();
        cout << YELLOW << "Sequence Mode Time: " << BOLD << seq_time << " ms" << RESET << endl;
    }

    // Run OMP mode only if mode is "OMP" or "ALL"
    if (mode == "OMP" || (mode == "ALL" && rank == 0)) {
#ifdef _OPENMP
        cout << YELLOW << "\nRunning OpenMP mode" << BOLD << "..." << RESET << endl;
        SimplexOMP simplexOMP(A, B, C);
        start = high_resolution_clock::now();
        ompR = simplexOMP.CalculateSimplex();
        end = high_resolution_clock::now();
        omp_time = duration_cast<milliseconds>(end - start).count();
        cout << YELLOW << "OpenMP Mode Time: " << BOLD << omp_time << " ms" << RESET << endl;
#else
        cerr << RED << "OpenMP not supported." << RESET << endl;
#endif
    }

    // Run CUDA mode only if mode is "CUDA" or "ALL"
    if (mode == "CUDA" || (mode == "ALL" && rank == 0)) {
#ifdef USE_CUDA
        cout << YELLOW << "\nRunning CUDA mode" << BOLD << "..." << RESET << endl;
        SimplexCUDA simplexCUDA(A, B, C);
        start = high_resolution_clock::now();
        cudaR = simplexCUDA.CalculateSimplex();
        end = high_resolution_clock::now();
        cuda_time = duration_cast<milliseconds>(end - start).count();
        cout << YELLOW << "CUDA Mode Time: " << BOLD << cuda_time << " ms" << RESET << endl;
#else
        cerr << RED << "CUDA not supported." << RESET << endl;
#endif
    }

    // Run MPI mode only if mode is "MPI" or "ALL"
    if (mode == "MPI" || mode == "ALL") {
#ifdef USE_MPI
        if (rank == 0) {
            cout << YELLOW << "\nRunning MPI mode" << BOLD << "..." << RESET << endl;
        }
        SimplexMPI simplexMPI(A, B, C);
        start = high_resolution_clock::now();
        mpiR = simplexMPI.CalculateSimplex();
        end = high_resolution_clock::now();
        mpi_time = duration_cast<milliseconds>(end - start).count();
        if (rank == 0) {
            cout << YELLOW << "MPI Mode Time: " << BOLD << mpi_time << " ms" << RESET << endl;
        }
        MPI_Finalize();
#else
        cerr << RED << "MPI not supported." << RESET << endl;
#endif
    }

    // Make a value no equal 0
    seq_time = (seq_time <= 0) ? 1 : seq_time;
    omp_time = (omp_time <= 0) ? 1 : omp_time;
    cuda_time = (cuda_time <= 0) ? 1 : cuda_time;
    mpi_time = (mpi_time <= 0) ? 1 : mpi_time;

    // Display comparison of the original (sequence) and modified (OMP, CUDA, MPI) run times
    if (((mode == "MPI" || mode == "ALL") && rank == 0) || rank == -1) {
        cout << YELLOW << "\n====================" << BOLD << " Performance Comparison Table " << RESET << YELLOW << "====================" << RESET << endl;
        cout << BOLD << YELLOW << setw(15) << "Mode"
            << setw(15) << "Time (ms)"
            << setw(25) << "Speed-up (SEQ/Mode)"
            << setw(15) << "Same with SEQ" << RESET
            << endl;
        cout << YELLOW << string(70, '-') << RESET << endl;
    }

    if (mode == "SEQUENCE" || (mode == "ALL" && rank == 0)) {
        cout << fixed << setprecision(2);
        cout << BOLD << CYAN << setw(15) << "Sequence"
            << setw(15) << seq_time
            << GREEN << setw(23) << floatToString(1) << "x" << RESET
            << endl;
    }
#ifdef _OPENMP
    if (mode == "OMP" || (mode == "ALL" && rank == 0)) {
        float count = seq_time / omp_time;
        cout << fixed << setprecision(2);
        cout << BOLD << CYAN << setw(15) << "OpenMP"
            << setw(15) << omp_time << RESET
            << ((count < 2) ? "" : BOLD) << setw(22) << ((count < 1) ? RED : GREEN) << floatToString(count) << "x" << RESET
            << setw(15) << (areVectorsEqual(seqR, ompR) ? GREEN + "SAME" + RESET : RED + "NO" + RESET)
            << endl;
    }
#endif

#ifdef USE_CUDA
    if (mode == "CUDA" || (mode == "ALL" && rank == 0)) {
        float count = seq_time / cuda_time;
        cout << fixed << setprecision(2);
        cout << BOLD << CYAN << setw(15) << "CUDA"
            << setw(15) << cuda_time << RESET
            << ((count < 2) ? "" : BOLD) << setw(22) << ((count < 1) ? RED : GREEN) << floatToString(count) << "x" << RESET
            << setw(15) << (areVectorsEqual(seqR, cudaR) ? GREEN + "SAME" + RESET : RED + "NO" + RESET)
            << endl;
    }

#endif

#ifdef USE_MPI
    if ((mode == "MPI" || mode == "ALL") && rank == 0) {
        float count = seq_time / mpi_time;
        cout << fixed << setprecision(2);
        cout << BOLD << CYAN << setw(15) << "MPI"
            << setw(15) << mpi_time << RESET
            << ((count < 2) ? "" : BOLD) << setw(22) << ((count < 1) ? RED : GREEN) << floatToString(count) << "x" << RESET
            << setw(15) << (areVectorsEqual(seqR, mpiR) ? GREEN + "SAME" + RESET : RED + "NO" + RESET)
            << endl;
    }
#endif

    if(rank == 0)
        cout << YELLOW << string(70, '-') << RESET << endl;


    return 0;
}