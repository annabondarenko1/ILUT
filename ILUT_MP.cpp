#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <omp.h>

using namespace std;

struct CSRMatrix {
    vector<double> values;
    vector<int> col_indices;
    vector<int> row_ptr;
};

CSRMatrix generateRandomMatrix(int n, int sparsity) {
    CSRMatrix A;
    A.row_ptr.push_back(0);
    srand(time(0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (rand() % 100 < sparsity) {
                double val = (rand() % 100) / 10.0;
                A.values.push_back(val);
                A.col_indices.push_back(j);
            }
        }
        A.row_ptr.push_back(A.values.size());
    }
    return A;
}

double rowNorm(const CSRMatrix &A, int row) {
    double norm = 0.0;
    for (int j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
        norm += A.values[j] * A.values[j];
    }
    return sqrt(norm);
}

CSRMatrix LUdecomposition(const CSRMatrix &A, int n, double tau) {
    CSRMatrix L, U;
    L.row_ptr.push_back(0);
    U.row_ptr.push_back(0);
    
    vector<double> rowU(n, 0.0);
    vector<double> rowL(n, 0.0);

    #pragma omp parallel for private(rowU, rowL) shared(L, U)
    for (int i = 0; i < n; ++i) {
        fill(rowU.begin(), rowU.end(), 0.0);
        fill(rowL.begin(), rowL.end(), 0.0);
        
        double norm_i = rowNorm(A, i);
        double tau_i = tau * norm_i;
        
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_indices[j];
            if (col < i) {
                rowL[col] = A.values[j];
                for (int k = 0; k < col; ++k) {
                    rowL[col] -= L.values[L.row_ptr[i] + k] * U.values[U.row_ptr[k] + col];
                }
                rowL[col] /= U.values[U.row_ptr[col] + col];
                if (abs(rowL[col]) < tau_i) {
                    rowL[col] = 0.0;
                }
            } else {
                rowU[col] = A.values[j];
                for (int k = 0; k < i; ++k) {
                    rowU[col] -= L.values[L.row_ptr[i] + k] * U.values[U.row_ptr[k] + col];
                }
                if (abs(rowU[col]) < tau_i) {
                    rowU[col] = 0.0;
                }
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < i; ++j) {
                if (rowL[j] != 0.0) {
                    L.values.push_back(rowL[j]);
                    L.col_indices.push_back(j);
                }
            }
            L.values.push_back(1.0);
            L.col_indices.push_back(i);
            L.row_ptr.push_back(L.values.size());
            
            for (int j = i; j < n; ++j) {
                if (rowU[j] != 0.0) {
                    U.values.push_back(rowU[j]);
                    U.col_indices.push_back(j);
                }
            }
            U.row_ptr.push_back(U.values.size());
        }
    }
    
    return L;
}

void printCSRMatrix(const CSRMatrix &A, int n) {
    cout << "Values: ";
    for (double v : A.values) {
        cout << v << " ";
    }
    cout << "\nColumn indices: ";
    for (int ci : A.col_indices) {
        cout << ci << " ";
    }
    cout << "\nRow pointer: ";
    for (int rp : A.row_ptr) {
        cout << rp << " ";
    }
    cout << endl;
}

int main() {
    int n = 5;  // размер матрицы
    int sparsity = 50; // вероятность ненулевого элемента (в процентах)
    double tau = 0.1; // относительный допуск

    CSRMatrix A = generateRandomMatrix(n, sparsity);
    cout << "Matrix A (in CSR format):" << endl;
    printCSRMatrix(A, n);

    CSRMatrix L = LUdecomposition(A, n, tau);

    cout << "\nMatrix L (in CSR format):" << endl;
    printCSRMatrix(L, n);

    return 0;
}
