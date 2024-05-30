#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

struct CSRMatrix {
    vector<double> values;
    vector<int> columns;
    vector<int> rowIndex;
};

// Функция для создания CSR-матрицы
CSRMatrix generateRandomCSRMatrix(int size, double density = 0.5, double minVal = -10, double maxVal = 10) {
    CSRMatrix csr;
    csr.rowIndex.push_back(0);
    srand(time(0));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if ((rand() / (double)RAND_MAX) < density) {
                double value = minVal + (maxVal - minVal) * (rand() / (double)RAND_MAX);
                csr.values.push_back(value);
                csr.columns.push_back(j);
            }
        }
        csr.rowIndex.push_back(csr.values.size());
    }

    return csr;
}

// Функция для получения строки из CSR-матрицы
vector<double> getRow(const CSRMatrix& matrix, int row) {
    vector<double> result(matrix.rowIndex.size() - 1, 0.0);
    for (int idx = matrix.rowIndex[row]; idx < matrix.rowIndex[row + 1]; ++idx) {
        result[matrix.columns[idx]] = matrix.values[idx];
    }
    return result;
}

// Функция для вычисления нормы строки
double rowNorm(const vector<double>& row) {
    double norm = 0.0;
    for (double value : row) {
        norm += fabs(value);
    }
    return norm;
}

// Функция для LU-разложения с учетом относительной погрешности в формате CSR
void LUDecompositionCSR(const CSRMatrix& A, CSRMatrix& L, CSRMatrix& U, double tau) {
    int n = A.rowIndex.size() - 1;
    vector<unordered_map<int, double>> lRows(n), uRows(n);

    for (int i = 0; i < n; ++i) {
        vector<double> w = getRow(A, i);
        double norm_i = rowNorm(w);
        double tau_i = tau * norm_i;

        for (int k = 0; k < i; ++k) {
            if (uRows[k].count(k) > 0 && uRows[k][k] != 0) {
                w[k] /= uRows[k][k];
                if (fabs(w[k]) < tau_i) {
                    w[k] = 0;
                }
                if (w[k] != 0) {
                    for (auto& pair : uRows[k]) {
                        int col = pair.first;
                        double value = pair.second;
                        w[col] -= w[k] * value;
                    }
                }
            } else {
                w[k] = 0; // Avoid division by zero
            }
        }

        for (int j = 0; j < i; ++j) {
            if (w[j] != 0) {
                lRows[i][j] = w[j];
            }
        }
        lRows[i][i] = 1.0; // Диагональные элементы L равны 1

        for (int j = i; j < n; ++j) {
            if (w[j] != 0) {
                uRows[i][j] = w[j];
            }
        }
    }

    // Преобразование lRows и uRows в CSR формат
    L.rowIndex.push_back(0);
    U.rowIndex.push_back(0);

    for (int i = 0; i < n; ++i) {
        for (auto& pair : lRows[i]) {
            L.values.push_back(pair.second);
            L.columns.push_back(pair.first);
        }
        L.rowIndex.push_back(L.values.size());

        for (auto& pair : uRows[i]) {
            U.values.push_back(pair.second);
            U.columns.push_back(pair.first);
        }
        U.rowIndex.push_back(U.values.size());
    }
}

// Функция для преобразования CSR-матрицы в обычный формат
vector<vector<double>> convertCSRToDense(const CSRMatrix& matrix) {
    int n = matrix.rowIndex.size() - 1;
    vector<vector<double>> denseMatrix(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int idx = matrix.rowIndex[i]; idx < matrix.rowIndex[i + 1]; ++idx) {
            denseMatrix[i][matrix.columns[idx]] = matrix.values[idx];
        }
    }

    return denseMatrix;
}

// Функция для вывода обычной матрицы
void printDenseMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double value : row) {
            cout << value << " ";
        }
        cout << endl;
    }
}

int main() {
    int size = 5; // Размер матрицы
    std::cout << "N: ";
    std::cin >> size;
    double tau = 1e-5;
    CSRMatrix csrA = generateRandomCSRMatrix(size);
    CSRMatrix L, U;
    auto start = std::chrono::high_resolution_clock::now();
    LUDecompositionCSR(csrA, L, U, tau);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Время работы программы: " << duration.count() << " миллисекунд" << std::endl;
    vector<vector<double>> denseA = convertCSRToDense(csrA);
    vector<vector<double>> denseL = convertCSRToDense(L);
    vector<vector<double>> denseU = convertCSRToDense(U);

    // cout << "Matrix A in dense format:" << endl;
    // printDenseMatrix(denseA);

    // cout << "Matrix L in dense format:" << endl;
    // printDenseMatrix(denseL);

    // cout << "Matrix U in dense format:" << endl;
    // printDenseMatrix(denseU);

    return 0;
}
