#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// Структура для представления матрицы в формате CSR
struct CSRMatrix {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
    int n; // Размер матрицы (предполагаем, что она квадратная)

    CSRMatrix(int size) : n(size) {
        row_ptr.resize(n + 1, 0);
    }

    // Функция для добавления значения в CSR матрицу
    void addValue(int row, int col, double value) {
        if (row < 0 || row >= n || col < 0 || col >= n) {
            std::cerr << "Invalid index for adding value: (" << row << "," << col << ")" << std::endl;
            return;
        }
        row_ptr[row + 1]++;
        col_idx.push_back(col);
        values.push_back(value);
    }

    // Завершение формирования CSR матрицы
    void finalize() {
        for (int i = 1; i <= n; ++i) {
            row_ptr[i] += row_ptr[i - 1];
        }
    }
};

// Функция для генерации случайной матрицы в формате CSR
void generateRandomCSRMatrix(CSRMatrix &A, int density, double maxValue) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) {
            if (std::rand() % 100 < density) { // Density процентов ненулевых элементов
                double value = static_cast<double>(std::rand()) / RAND_MAX * maxValue;
                A.addValue(i, j, value);
            }
        }
    }
    A.finalize();
}

// Функция для поиска индекса в CSR массиве
int findIndex(const CSRMatrix &matrix, int row, int col) {
    for (int j = matrix.row_ptr[row]; j < matrix.row_ptr[row + 1]; ++j) {
        if (matrix.col_idx[j] == col) {
            return j;
        }
    }
    return -1;
}

// ILU(tau) разложение
void ILU_tau(CSRMatrix &A, CSRMatrix &L, CSRMatrix &U, double tau) {
    std::vector<double> diag(A.n, 0.0);

    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_idx[j];
            double value = A.values[j];

            if (col < i) { // Заполнение L
                double sum = value;
                for (int k = A.row_ptr[i]; k < j; ++k) {
                    int l_idx = findIndex(L, i, A.col_idx[k]);
                    int u_idx = findIndex(U, col, A.col_idx[k]);
                    if (l_idx != -1 && u_idx != -1) {
                        sum -= L.values[l_idx] * U.values[u_idx];
                    }
                }
                if (diag[col] != 0) {
                    sum /= diag[col];
                }
                if (std::abs(sum) >= tau) {
                    L.addValue(i, col, sum);
                }
            } else if (col == i) { // Диагональ
                double sum = value;
                for (int k = A.row_ptr[i]; k < j; ++k) {
                    int l_idx = findIndex(L, i, A.col_idx[k]);
                    int u_idx = findIndex(U, col, A.col_idx[k]);
                    if (l_idx != -1 && u_idx != -1) {
                        sum -= L.values[l_idx] * U.values[u_idx];
                    }
                }
                diag[i] = sum;
                if (std::abs(sum) >= tau) {
                    U.addValue(i, col, sum);
                }
            } else { // Заполнение U
                double sum = value;
                for (int k = A.row_ptr[i]; k < j; ++k) {
                    int l_idx = findIndex(L, i, A.col_idx[k]);
                    int u_idx = findIndex(U, col, A.col_idx[k]);
                    if (l_idx != -1 && u_idx != -1) {
                        sum -= L.values[l_idx] * U.values[u_idx];
                    }
                }
                if (std::abs(sum) >= tau) {
                    U.addValue(i, col, sum);
                }
            }
        }
    }

    L.finalize();
    U.finalize();
}

// Функция для преобразования CSR матрицы в плотную матрицу
std::vector<std::vector<double>> convertCSRToDense(const CSRMatrix &A) {
    std::vector<std::vector<double>> dense(A.n, std::vector<double>(A.n, 0.0));
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense[i][A.col_idx[j]] = A.values[j];
        }
    }
    return dense;
}

// Функция для вывода плотной матрицы
void printDenseMatrix(const std::vector<std::vector<double>> &matrix) {
    for (const auto &row : matrix) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}



int main() {
    int n = 4;
    int density = 30; // Процент ненулевых элементов
    double maxValue = 10.0; // Максимальное значение элементов

    CSRMatrix A(n);
    generateRandomCSRMatrix(A, density, maxValue);

    CSRMatrix L(n);
    CSRMatrix U(n);

    double tau = 0.01;
    ILU_tau(A, L, U, tau);

    std::cout << "Generated matrix A:" << std::endl;
    auto denseA = convertCSRToDense(A);
    printDenseMatrix(denseA);

    std::cout << "L matrix:" << std::endl;
    auto denseL = convertCSRToDense(L);
    printDenseMatrix(denseL);

    std::cout << "U matrix:" << std::endl;
    auto denseU = convertCSRToDense(U);
    printDenseMatrix(denseU);

    return 0;
}
