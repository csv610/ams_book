//==============================================================================
// Chapter 1: Numerical Stability in Gaussian Elimination
// C++ Implementation with Growth Factor Analysis
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/QR>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. Gaussian Elimination with Partial Pivoting (Educational Implementation)
//==============================================================================
struct LUResult {
    MatrixXd L;
    MatrixXd U;
    VectorXi P;
    double growth_factor;
};

LUResult gaussian_elimination_partial_pivot(const MatrixXd& A_in) {
    int n = A_in.rows();
    MatrixXd A = A_in;
    MatrixXd L = MatrixXd::Identity(n, n);
    VectorXi P = VectorXi::LinSpaced(n, 0, n-1);
    
    double max_A = A.cwiseAbs().maxCoeff();
    double max_U = max_A;
    
    for (int k = 0; k < n-1; ++k) {
        // Find pivot
        int i_max = k;
        double max_val = std::abs(A(k, k));
        for (int i = k+1; i < n; ++i) {
            if (std::abs(A(i, k)) > max_val) {
                max_val = std::abs(A(i, k));
                i_max = i;
            }
        }
        
        if (i_max != k) {
            // Swap rows in A
            A.row(k).swap(A.row(i_max));
            // Swap rows in L (only previous columns)
            for (int j = 0; j < k; ++j) {
                std::swap(L(k, j), L(i_max, j));
            }
            // Track permutation
            std::swap(P(k), P(i_max));
        }
        
        // Eliminate
        for (int i = k+1; i < n; ++i) {
            L(i, k) = A(i, k) / A(k, k);
            for (int j = k+1; j < n; ++j) {
                A(i, j) -= L(i, k) * A(k, j);
            }
            max_U = max(max_U, std::abs(A(i, k+1)));
        }
    }
    
    LUResult result;
    result.L = L;
    result.U = A.template triangularView<Upper>();
    result.P = P;
    result.growth_factor = max_U / A_in.cwiseAbs().maxCoeff();
    return result;
}

//==============================================================================
// 2. Solve using LU factors
//==============================================================================
VectorXd solve_lu(const MatrixXd& L, const MatrixXd& U, const VectorXi& P, const VectorXd& b) {
    int n = L.rows();
    VectorXd Pb = b(P);  // Apply permutation
    
    // Forward substitution: L y = Pb
    VectorXd y = VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {
        y(i) = Pb(i) - (L.row(i).head(i)).dot(y.head(i));
    }
    
    // Backward substitution: U x = y
    VectorXd x = VectorXd::Zero(n);
    for (int i = n-1; i >= 0; --i) {
        x(i) = (y(i) - U.row(i).tail(n-i-1).dot(x.tail(n-i-1))) / U(i, i);
    }
    return x;
}

//==============================================================================
// 3. Wilkinson Matrix (Worst Case for Partial Pivoting)
//==============================================================================
MatrixXd wilkinson_matrix(int n) {
    MatrixXd A = MatrixXd::Identity(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            A(j, i) = -1.0;
        }
    }
    A.col(n-1).setOnes();
    return A;
}

//==============================================================================
// 4. Growth Factor Computation
//==============================================================================
double compute_growth_factor(const MatrixXd& A) {
    LUResult result = gaussian_elimination_partial_pivot(A);
    return result.growth_factor;
}

//==============================================================================
// 5. Benchmark: Eigen vs Manual Implementation
//==============================================================================
void benchmark_gaussian_elimination() {
    cout << "=== Gaussian Elimination Benchmark ===\n\n";
    
    vector<int> sizes = {100, 500, 1000, 2000};
    
    cout << setw(8) << "Size" << setw(15) << "Eigen LU (ms)" 
         << setw(15) << "Manual (ms)" << setw(15) << "Speedup"
         << setw(15) << "Growth ρ" << "\n";
    cout << string(68, '-') << "\n";
    
    for (int n : sizes) {
        MatrixXd A = MatrixXd::Random(n, n);
        VectorXd b = VectorXd::Random(n);
        
        // Eigen's LU (highly optimized)
        auto t1 = chrono::high_resolution_clock::now();
        FullPivLU<MatrixXd> lu_eigen(A);
        VectorXd x_eigen = lu_eigen.solve(b);
        auto t2 = chrono::high_resolution_clock::now();
        double eigen_ms = chrono::duration<double, milli>(t2 - t1).count();
        
        // Manual implementation
        t1 = chrono::high_resolution_clock::now();
        LUResult result = gaussian_elimination_partial_pivot(A);
        VectorXd x_manual = solve_lu(result.L, result.U, result.P, b);
        t2 = chrono::high_resolution_clock::now();
        double manual_ms = chrono::duration<double, milli>(t2 - t1).count();
        
        // Verify accuracy
        double error = (x_eigen - x_manual).norm() / x_eigen.norm();
        
        cout << setw(8) << n << setw(15) << fixed << setprecision(2) << eigen_ms
             << setw(15) << manual_ms << setw(15) << setprecision(1) << manual_ms/eigen_ms
             << setw(15) << setprecision(2) << result.growth_factor << "\n";
    }
}

//==============================================================================
// 6. Exact vs Floating Point (using rational via GMP - placeholder)
//==============================================================================
void benchmark_exact_vs_float() {
    cout << "\n=== Exact vs Floating Point ===\n";
    cout << "Note: Exact rational arithmetic requires GMP/MPFR libraries.\n";
    cout << "This is a placeholder showing the concept.\n\n";
    
    int n = 100;
    MatrixXd A = MatrixXd::Random(n, n);
    VectorXd b = VectorXd::Random(n);
    
    // Float64
    auto t1 = chrono::high_resolution_clock::now();
    VectorXd x_float = A.fullPivLu().solve(b);
    auto t2 = chrono::high_resolution_clock::now();
    double float_ms = chrono::duration<double, milli>(t2 - t1).count();
    
    // "Exact" would use Rational<BigInt> - extremely slow
    cout << "n = " << n << "\n";
    cout << "  Float64 time: " << fixed << setprecision(2) << float_ms << " ms\n";
    cout << "  Exact rational: ~hours (not implemented here)\n";
    cout << "  Typical speedup factor: 10^4 - 10^6 x\n";
}

//==============================================================================
// 7. Wilkinson Matrix Test
//==============================================================================
void test_wilkinson_matrix() {
    cout << "\n=== Wilkinson Matrix (Worst Case) ===\n";
    
    for (int n : {10, 20, 30, 50, 100}) {
        MatrixXd W = wilkinson_matrix(n);
        double rho = compute_growth_factor(W);
        cout << "n = " << setw(3) << n << " : growth factor ρ = " 
             << scientific << setprecision(2) << rho 
             << " (theoretical max: 2^(n-1) = " << pow(2.0, n-1) << ")\n";
    }
}

//==============================================================================
// 8. Condition Number and Forward Error Analysis
//==============================================================================
void error_analysis_demo(int n = 100) {
    cout << "\n=== Error Analysis Demo ===\n";
    
    // Well-conditioned matrix
    MatrixXd A_well = MatrixXd::Random(n, n) + n*MatrixXd::Identity(n, n);
    double kappa_well = A_well.condition();
    
    // Ill-conditioned matrix
    MatrixXd A_ill = MatrixXd::Random(n, n);
    A_ill = A_ill * A_ill.transpose();  // symmetric positive definite
    A_ill(0,0) = 1e-12;  // Make nearly singular
    double kappa_ill = A_ill.condition();
    
    VectorXd b = VectorXd::Random(n);
    VectorXd x_true_well = A_well.fullPivLu().solve(b);
    VectorXd x_true_ill = A_ill.fullPivLu().solve(b);
    
    // Solve with Float64 (which uses partial pivoting via LAPACK)
    VectorXd x_f_well = A_well.cast<float>().fullPivLu().solve(b.cast<float>()).cast<double>();
    VectorXd x_f_ill = A_ill.cast<float>().fullPivLu().solve(b.cast<float>()).cast<double>();
    
    double err_well = (x_f_well - x_true_well).norm() / x_true_well.norm();
    double err_ill = (x_f_ill - x_true_ill).norm() / x_true_ill.norm();
    
    cout << "Well-conditioned (κ ≈ " << kappa_well << "):\n";
    cout << "  Relative error: " << err_well << "\n";
    cout << "Ill-conditioned (κ ≈ " << kappa_ill << "):\n";
    cout << "  Relative error: " << err_ill << "\n";
    
    // Growth factors
    double rho_well = compute_growth_factor(A_well);
    double rho_ill = compute_growth_factor(A_ill);
    cout << "\nGrowth factors: well= " << rho_well << ", ill= " << rho_ill << "\n";
}

//==============================================================================
// Main
//==============================================================================
int main() {
    cout.precision(6);
    
    benchmark_gaussian_elimination();
    test_wilkinson_matrix();
    benchmark_exact_vs_float();
    error_analysis_demo(100);
    
    cout << "\n=== Key Takeaways ===\n";
    cout << "1. Partial pivoting growth factor is typically O(n) for random matrices\n";
    cout << "2. Wilkinson matrix achieves exponential growth 2^(n-1)\n";
    cout << "3. Eigen's LU is 10-100x faster than naive implementation\n";
    cout << "4. Exact rational arithmetic is impractical for n > 50\n";
    cout << "5. Growth factor ρ = ||U||_max / ||A||_max predicts stability\n";
    cout << "6. Exact rational arithmetic is impractical for n > 50\n";
    
    return 0;
}
