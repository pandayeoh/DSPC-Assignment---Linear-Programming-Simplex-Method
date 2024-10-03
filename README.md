# DSPC-Assignment---Linear-Programming-Simplex-Method
## Use PowerShell
Run the following after opening PowerShell
```
cd x64/Debug
.\run_simplex.bat <dataset> <mode> <isPrint>
```

## Dataset Example - Dataset to run
* test
* test2

PS: can go x64/Debug/Dataset to add new .csv file (Need follow format)

## Mode Example - Method to use
* SEQUENCE
* OMP
* CUDA
* MPI
* ALL

## isPrint Example - Whether to display the solution
* YES
* NO

## Example Code
```
.\run_simplex.bat test OMP NO
```

## Preset:
### If no argument is entered, the syntax by default is
```
.\run_simplex.bat test ALL NO
```

## Bulid Dataset
* The first column and row of the dataset only store the variable name
* The data starts from the B2 column
* The last column shows the constraint of each row which is asummed to be less than or equal to
* The last row shows the maximization equation, the value in the last row must be negative or 0
* The last cell which is the intersection of the last column and row must be 0 which is the slack variables
* All variables are assumed to be non negative

### Equation Example
```
7x1 + 6x2
2x1 + 4x2 <= 16
3x1 + 2x2 <= 12
x1,x2 > 0
```

### Dataset in CSV Example
```
name    | columnName | columnName | lastColumn
rowName | 2          | 4          | 16
rowName | 3          | 2          | 12
lastRow | -7         | -6         | 0
```
