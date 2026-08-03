//==============================================================================
// Chapter 2: Algorithmic Fewnomial Theory
// C++ Implementation with Tropical Geometry
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <tuple>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. Tropical Arithmetic
//==============================================================================
namespace Tropical {
    const double INF = numeric_limits<double>::infinity();
    const double NEG_INF = -INF;
    
    // Tropical addition = max
    double add(double a, double b) { return max(a, b); }
    // Tropical multiplication = addition
    double mul(double a, double b) { return a + b; }
    // Tropical division = subtraction
    double div(double a, double b) { return a - b; }
    
    // Tropical polynomial evaluation at point x
    double eval_poly(const vector<pair<VectorXi, double>>& terms, const VectorXd& x) {
        double result = NEG_INF;
        for (auto& [alpha, c] : terms) {
            double val = c;
            for (int i = 0; i < alpha.size(); ++i) {
                val += alpha(i) * x(i);
            }
            result = max(result, val);
        }
        return result;
    }
}

//==============================================================================
// 2. Fewnomial Representation
//==============================================================================
struct Fewnomial {
    // Terms: (exponent vector, coefficient)
    vector<pair<VectorXi, double>> terms;
    int n_vars;
    
    Fewnomial(int n) : n_vars(n) {}
    
    void add_term(const VectorXi& alpha, double c) {
        if (std::abs(c) > 1e-14) terms.push_back({alpha, c});
    }
    
    // Evaluate at point x (standard arithmetic)
    double eval(const VectorXd& x) const {
        double result = 0.0;
        for (auto& [alpha, c] : terms) {
            double val = c;
            for (int i = 0; i < n_vars; ++i) {
                val *= pow(x(i), alpha(i));
            }
            result += val;
        }
        return result;
    }
    
    // Evaluate tropical polynomial (for initial guess)
    double eval_tropical(const VectorXd& x) const {
        double result = Tropical::NEG_INF;
        for (auto& [alpha, c] : terms) {
            double val = log(std::abs(c));
            for (int i = 0; i < n_vars; ++i) {
                val += alpha(i) * x(i);
            }
            result = max(result, val);
        }
        return result;
    }
    
    // Support set
    MatrixXi support() const {
        MatrixXi A(terms.size(), n_vars);
        for (size_t i = 0; i < terms.size(); ++i) {
            A.row(i) = terms[i].first.transpose();
        }
        return A;
    }
    
    int num_terms() const { return terms.size(); }
};

//==============================================================================
// 3. Tropical Variety / Positive Tropical Variety
//==============================================================================
struct TropicalPolyhedron {
    // Represented as intersection of halfspaces: A x <= b (in tropical sense)
    // For a k-nomial, Trop+(f) is union of at most C(k,2) polyhedral cones
    
    struct Cone {
        MatrixXd A_ub;  // Inequalities: A_ub * x <= b_ub
        VectorXd b_ub;
        MatrixXd A_eq;  // Equalities: A_eq * x == b_eq
        VectorXd b_eq;
    };
    
    vector<Cone> cones;
    
    static TropicalPolyhedron from_fewnomial(const Fewnomial& f) {
        TropicalPolyhedron T;
        int k = f.terms.size();
        
        for (int i = 0; i < k; ++i) {
            for (int j = i+1; j < k; ++j) {
                // Equality of two terms: alpha_i^T x + log|c_i| = alpha_j^T x + log|c_j|
                VectorXi alpha_i = f.terms[i].first;
                VectorXi alpha_j = f.terms[j].first;
                double c_i = f.terms[i].second;
                double c_j = f.terms[j].second;
                
                VectorXd A_eq = (alpha_i - alpha_j).cast<double>();
                double b_eq = log(std::abs(c_j)) - log(std::abs(c_i));
                
                // Dominance over other terms: alpha_i^T x + log|c_i| >= alpha_l^T x + log|c_l|
                int n = f.n_vars;
                MatrixXd A_ub(k-2, n);
                VectorXd b_ub(k-2);
                int row = 0;
                for (int l = 0; l < k; ++l) {
                    if (l == i || l == j) continue;
                    VectorXi alpha_l = f.terms[l].first;
                    double c_l = f.terms[l].second;
                    A_ub.row(row) = (alpha_i - alpha_l).cast<double>();
                    b_ub(row) = log(std::abs(c_l)) - log(std::abs(c_i));
                    row++;
                }
                
                TropicalPolyhedron::Cone cone;
                cone.A_ub = A_ub;
                cone.b_ub = b_ub;
                cone.A_eq = A_eq.transpose();
                cone.b_eq.resize(1);
                cone.b_eq(0) = b_eq;
                T.cones.push_back(cone);
            }
        }
        return T;
    }
    
    // Find a feasible point in the tropical variety
    optional<VectorXd> find_feasible_point() const {
        for (const auto& cone : cones) {
            // Solve linear feasibility: A_ub x <= b_ub, A_eq x = b_eq
            // Using Eigen's built-in or external LP solver
            // Placeholder: return center of first cone
            VectorXd x = cone.A_eq.colPivHouseholderQr().solve(cone.b_eq);
            if ((cone.A_ub * x).array().maxCoeff() <= cone.b_ub.array().maxCoeff() + 1e-9) {
                return x;
            }
        }
        return nullopt;
    }
};

//==============================================================================
// 4. VT Codes for Single-Deletion Correction
//==============================================================================
class VTCode {
    int n;      // codeword length
    int a;      // parameter in 0..n
    
public:
    VTCode(int n_, int a_) : n(n_), a(a_) {}
    
    // Systematic encoding: find parity bit position and value
    // such that sum_{i=1}^n i * x_i ≡ a (mod n+1)
    vector<int> encode(const vector<int>& message) {
        // message has length n-1, we insert one parity bit
        int m = n - 1;
        if (message.size() != m) {
            throw runtime_error("Message length must be n-1");
        }
        
        // Try inserting parity at each position
        for (int pos = 0; pos <= m; ++pos) {
            vector<int> codeword(n);
            // Fill codeword with message bits, leaving pos for parity
            int msg_idx = 0;
            for (int i = 0; i < n; ++i) {
                if (i == pos) continue;
                codeword[i] = message[msg_idx++];
            }
            
            // Find parity bit value (0 or 1) that satisfies VT condition
            for (int parity = 0; parity <= 1; ++parity) {
                codeword[pos] = parity;
                if (check_vt(codeword)) {
                    return codeword;
                }
            }
        }
        throw runtime_error("No valid codeword found");
    }
    
    // Check if codeword is in VT_a(n)
    bool check_vt(const vector<int>& x) {
        long long sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += (i+1) * x[i];
        }
        return (sum % (n+1)) == a;
    }
    
    // Decode received word with single deletion
    vector<int> decode(const vector<int>& y) {
        // y has length n-1
        if (y.size() != n-1) {
            throw runtime_error("Received word must have length n-1");
        }
        
        int s = 0;
        for (int i = 0; i < n-1; ++i) {
            s += (i+1) * y[i];
        }
        s %= (n+1);
        
        // The syndrome s tells us the position and value of deleted bit
        // Full implementation requires careful analysis
        // Placeholder: return original (in practice, use Levenshtein's algorithm)
        return vector<int>(n, 0);
    }
};

//==============================================================================
// 5. Newton Homotopy for Fewnomial Systems
//==============================================================================
struct NewtonHomotopy {
    // System of fewnomials F(x) = 0
    // Start system G(x) = 0 (binomial, easy to solve)
    // Homotopy: H(x,t) = (1-t)G(x) + tF(x)
    
    struct System {
        vector<Fewnomial> equations;
        int n_vars;
    };
    
    System start_system;
    System target_system;
    
    NewtonHomotopy(const System& start, const System& target)
        : start_system(start), target_system(target) {}
    
    // Evaluate homotopy at (x,t)
    VectorXd eval(const VectorXd& x, double t) const {
        VectorXd result(start_system.n_vars);
        for (int i = 0; i < start_system.n_vars; ++i) {
            double g = start_system.equations[i].eval(x);
            double f = target_system.equations[i].eval(x);
            result(i) = (1-t)*g + t*f;
        }
        return result;
    }
    
    // Jacobian of homotopy
    MatrixXd jacobian(const VectorXd& x, double t) const {
        int n = start_system.n_vars;
        MatrixXd J(n, n);
        double eps = 1e-8;
        
        for (int j = 0; j < n; ++j) {
            VectorXd x_plus = x; x_plus(j) += eps;
            VectorXd x_minus = x; x_minus(j) -= eps;
            J.col(j) = (eval(x_plus, t) - eval(x_minus, t)) / (2*eps);
        }
        return J;
    }
    
    // Track path from t=0 to t=1 using predictor-corrector
    VectorXd track_path(const VectorXd& x0, double dt = 0.01) {
        VectorXd x = x0;
        for (double t = 0; t < 1.0; t += dt) {
            // Predictor: Euler step
            VectorXd F = eval(x, t);
            MatrixXd J = jacobian(x, t);
            VectorXd dx = -J.colPivHouseholderQr().solve(F);
            x += dx * dt;
            
            // Corrector: Newton iterations at new t
            for (int iter = 0; iter < 3; ++iter) {
                F = eval(x, t + dt);
                J = jacobian(x, t + dt);
                dx = -J.colPivHouseholderQr().solve(F);
                x += dx;
                if (dx.norm() < 1e-10) break;
            }
        }
        return x;
    }
};

//==============================================================================
// 6. Example: Phosphorylation Cycle
//==============================================================================
void phosphorylation_example() {
    cout << "\n=== Phosphorylation Cycle (Chemical Reaction Network) ===\n";
    
    // Simple phosphorylation: S + E <-> SE -> S_p + E, S_p + F <-> SF -> S + F
    // Steady states lead to fewnomial system
    
    // Variables: [S, E, SE, S_p, F, SF]
    // Simplified model yields fewnomials with 3-4 terms each
    
    Fewnomial f1(6);  // Conservation: S + SE + S_p + SF = S_tot
    f1.add_term(VectorXi::Zero(6), -1.0);  // constant
    f1.add_term((VectorXi() << 1,0,1,1,0,1).finished(), 1.0);
    
    Fewnomial f2(6);  // Conservation: E + SE = E_tot
    f2.add_term(VectorXi::Zero(6), -1.0);
    f2.add_term((VectorXi() << 0,1,1,0,0,0).finished(), 1.0);
    
    Fewnomial f3(6);  // k1*S*E = k2*SE (mass action)
    f3.add_term((VectorXi() << 1,1,0,0,0,0).finished(), 1.0);  // S*E
    f3.add_term((VectorXi() << 0,0,1,0,0,0).finished(), -0.5); // -k2/k1 * SE
    
    cout << "Fewnomial system for phosphorylation:\n";
    for (auto& f : {f1, f2, f3}) {
        cout << "  Terms: " << f.terms.size() << "\n";
        for (auto& [alpha, c] : f.terms) {
            cout << "    " << c << " * x^" << alpha.transpose() << "\n";
        }
    }
    
    // Tropical approximation for initial guess
    VectorXd x0(6); x0.setOnes();
    cout << "\nTropical evaluations at x=1:\n";
    for (auto& f : {f1, f2, f3}) {
        cout << "  " << f.eval_tropical(x0) << "\n";
    }
}

//==============================================================================
// Main
//==============================================================================
int main() {
    phosphorylation_example();
    
    // Test VT code
    cout << "\n=== VT Code Test ===\n";
    VTCode vt(10, 3);
    vector<int> msg = {1,0,1,1,0,1,0,0,1};
    auto cw = vt.encode(msg);
    cout << "Codeword: ";
    for (int b : cw) cout << b;
    cout << " (valid: " << vt.check_vt(cw) << ")\n";
    
    // Tropical variety
    cout << "\n=== Tropical Variety ===\n";
    Fewnomial f(2);
    f.add_term((VectorXi(2) << 2,0).finished(), 1.0);
    f.add_term((VectorXi(2) << 0,2).finished(), -1.0);
    f.add_term((VectorXi(2) << 1,1).finished(), 0.5);
    
    auto T = TropicalPolyhedron::from_fewnomial(f);
    cout << "Number of cones: " << T.cones.size() << "\n";
    if (auto pt = T.find_feasible_point()) {
        cout << "Feasible point: " << pt->transpose() << "\n";
    }
    
    return 0;
}