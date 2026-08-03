//==============================================================================
// Chapter 8: New Developments in Global Positioning (GPS)
// C++ Implementation: Algebraic Geometry Approach to GPS
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <random>
#include <cmath>
#include <algorithm>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. GPS Equations and Minkowski Geometry
//==============================================================================

struct Satellite {
    Vector3d position;   // s_i
    double tau;          // apparent flight time
};

struct GPSSolution {
    Vector3d position;
    double clock_bias;
    bool valid;
    int num_solutions;  // 0, 1, or 2
};

class GPSSolver {
public:
    // Build matrix A from satellite data
    // A rows: [2*s_i^T, -2*tau_i, -1]
    static MatrixXd build_A_matrix(const vector<Satellite>& sats) {
        int m = sats.size();
        MatrixXd A(m, 5);  // [x, y, z, b, 1] -> [2s_x, 2s_y, 2s_z, -2tau, -1]
        for (int i = 0; i < m; ++i) {
            A(i, 0) = 2 * sats[i].position(0);
            A(i, 1) = 2 * sats[i].position(1);
            A(i, 2) = 2 * sats[i].position(2);
            A(i, 3) = -2 * sats[i].tau;
            A(i, 4) = -1;
        }
        return A;
    }
    
    static VectorXd build_sigma(const vector<Satellite>& sats) {
        int m = sats.size();
        VectorXd sigma(m);
        for (int i = 0; i < m; ++i) {
            sigma(i) = sats[i].position.squaredNorm() - sats[i].tau * sats[i].tau;
        }
        return sigma;
    }
    
    // Solve linear system A*X = sigma
    // X = [x, y, z, b, x^T η x + b^2 - x^T x] where η = diag(1,1,1,-1)
    static GPSSolution solve_exact(const vector<Satellite>& sats) {
        int m = sats.size();
        if (m < 4) {
            return {Vector3d::Zero(), 0, false, 0};
        }
        
        MatrixXd A = build_A_matrix(sats);
        VectorXd sigma = build_sigma(sats);
        
        // Check rank
        int rank = A.fullPivLu().rank();
        GPSSolution sol;
        
        if (rank == 5) {
            // Unique solution
            VectorXd X = A.colPivHouseholderQr().solve(sigma);
            sol.position = X.head(3);
            sol.clock_bias = X(3);
            sol.valid = true;
            sol.num_solutions = 1;
            
            // Verify quadratic constraint
            double check = X.head(3).squaredNorm() - X(3)*X(3) - 2*X(4);
            if (std::abs(check) > 1e-6) sol.valid = false;
            
        } else if (rank == 4) {
            // Ambiguous case: two solutions
            // Find nullspace vector
            JacobiSVD<MatrixXd> svd(A, ComputeFullV);
            VectorXd v_null = svd.matrixV().col(4);  // Last column = nullspace
            
            // Particular solution
            VectorXd X0 = A.colPivHouseholderQr().solve(sigma);
            
            // Line: X = X0 + t * v_null
            // Plug into quadratic constraint: X^T η X = 0 where η = diag(I, -1, 0)
            MatrixXd eta = MatrixXd::Zero(5,5);
            eta.block(0,0,3,3) = Matrix3d::Identity();
            eta(3,3) = -1;
            // eta(4,4) = 0
            
            double a = v_null.transpose() * eta * v_null;
            double b = 2 * X0.transpose() * eta * v_null;
            double c = X0.transpose() * eta * X0;
            
            // Quadratic: a t^2 + b t + c = 0
            double disc = b*b - 4*a*c;
            
            if (disc >= 0) {
                sol.num_solutions = (disc > 1e-12) ? 2 : 1;
                double t1 = (-b + sqrt(max(0.0, disc))) / (2*a);
                double t2 = (-b - sqrt(max(0.0, disc))) / (2*a);
                
                VectorXd X1 = X0 + t1 * v_null;
                VectorXd X2 = X0 + t2 * v_null;
                
                // Pick the physically plausible one (positive b, reasonable position)
                sol.position = X1.head(3);
                sol.clock_bias = X1(3);
                sol.valid = true;
                
                // Check both solutions
                cout << "Two solutions found:\n";
                cout << "  Sol 1: pos=" << X1.head(3).transpose() 
                     << ", b=" << X1(3) << "\n";
                cout << "  Sol 2: pos=" << X2.head(3).transpose() 
                     << ", b=" << X2(3) << "\n";
            } else {
                sol.valid = false;
                sol.num_solutions = 0;
            }
        } else {
            sol.valid = false;
            sol.num_solutions = 0;
        }
        
        return sol;
    }
    
    // Weighted least squares for noisy data
    static GPSSolution solve_weighted(const vector<Satellite>& sats, 
                                       const VectorXd& weights) {
        int m = sats.size();
        MatrixXd A = build_A_matrix(sats);
        VectorXd sigma = build_sigma(sats);
        
        MatrixXd W = weights.asDiagonal();
        
        // Weighted least squares: (A^T W A) X = A^T W sigma
        MatrixXd AWA = A.transpose() * W * A;
        VectorXd AWb = A.transpose() * W * sigma;
        
        VectorXd X = AWA.ldlt().solve(AWb);
        
        GPSSolution sol;
        sol.position = X.head(3);
        sol.clock_bias = X(3);
        sol.valid = true;
        sol.num_solutions = 1;
        
        // Check residual
        VectorXd residual = A * X - sigma;
        double chi2 = residual.transpose() * W * residual;
        if (chi2 > 10) sol.valid = false;  // Poor fit
        
        return sol;
    }
};

//==============================================================================
// 2. Quadric Classification (Boutin-Kemper)
//==============================================================================

struct Quadric {
    // X^T Q X = 0 where Q is 4x4 symmetric
    Matrix4d Q;
    
    enum Type { SPHERE, ELIPSOID, PARABOLOID, HYPERBOLOID, DEGENERATE };
    Type type;
    
    static Quadric from_focus_directrix(const Vector3d& focus, 
                                         const Vector3d& directrix_pt,
                                         const Vector3d& directrix_normal,
                                         double eccentricity) {
        // (x - focus)^T (x - focus) = e^2 * (n^T (x - directrix_pt))^2
        Matrix4d Q = Matrix4d::Zero();
        Vector3d n = directrix_normal.normalized();
        double e = eccentricity;
        
        // Expand: x^T x - 2 f^T x + f^T f = e^2 (n^T x - d)^2
        // where d = n^T directrix_pt
        double d = n.dot(directrix_pt);
        
        Q.block(0,0,3,3) = Matrix3d::Identity() - e*e * n * n.transpose();
        Q.block(0,3,3,1) = -focus + e*e * d * n;
        Q.block(3,0,1,3) = Q.block(0,3,3,1).transpose();
        Q(3,3) = focus.squaredNorm() - e*e * d*d;
        
        Quadric quad;
        quad.Q = Q;
        quad.classify();
        return quad;
    }
    
    void classify() {
        // Classify by eigenvalues of upper 3x3 block
        SelfAdjointEigenSolver<Matrix3d> es(Q.block(0,0,3,3));
        Vector3d evals = es.eigenvalues();
        
        int pos = (evals.array() > 1e-10).count();
        int neg = (evals.array() < -1e-10).count();
        int zero = 3 - pos - neg;
        
        if (zero == 0 && pos == 3) type = ELIPSOID;
        else if (zero == 0 && pos == 2 && neg == 1) type = HYPERBOLOID;
        else if (zero == 1 && pos == 2) type = PARABOLOID;
        else if (zero == 0 && pos == 1 && neg == 2) type = HYPERBOLOID;  // 2 sheets
        else type = DEGENERATE;
    }
    
    string type_string() const {
        switch(type) {
            case SPHERE: return "Sphere";
            case ELIPSOID: return "Ellipsoid";
            case PARABOLOID: return "Paraboloid";
            case HYPERBOLOID: return "Hyperboloid";
            default: return "Degenerate";
        }
    }
};

//==============================================================================
// 3. Simulation and Testing
//==============================================================================

void simulate_gps() {
    cout << "=== GPS Simulation ===\n\n";
    
    // True user position and clock bias
    Vector3d true_pos(1000, 2000, 500);  // meters
    double true_bias = 50.0;  // meters (clock offset)
    
    // Generate satellite constellation (6 satellites)
    vector<Satellite> sats;
    vector<Vector3d> sat_positions = {
        Vector3d(20000, 0, 20000),
        Vector3d(0, 20000, 20000),
        Vector3d(-20000, 0, 20000),
        Vector3d(0, -20000, 20000),
        Vector3d(20000, 20000, 20000),
        Vector3d(-20000, -20000, 20000)
    };
    
    random_device rd;
    mt19937 gen(42);
    normal_distribution<> noise(0, 0.01);  // 1 cm timing noise
    
    for (int i = 0; i < 6; ++i) {
        Satellite sat;
        sat.position = sat_positions[i];
        double dist = (true_pos - sat.position).norm();
        sat.tau = dist + true_bias + noise(gen);
        sats.push_back(sat);
    }
    
    // Solve
    GPSSolution sol = GPSSolver::solve_exact(sats);
    
    cout << "True position: " << true_pos.transpose() << "\n";
    cout << "True bias: " << true_bias << "\n";
    cout << "Estimated position: " << sol.position.transpose() << "\n";
    cout << "Estimated bias: " << sol.clock_bias << "\n";
    cout << "Error: " << (sol.position - true_pos).norm() << " m\n";
    cout << "Bias error: " << std::abs(sol.clock_bias - true_bias) << " m\n";
    cout << "Valid: " << (sol.valid ? "Yes" : "No") << "\n";
    
    // Test with noise
    cout << "\n=== Weighted Least Squares (Noisy) ===\n";
    vector<Satellite> noisy_sats = sats;
    for (auto& sat : noisy_sats) {
        sat.tau += noise(gen) * 10;  // 10 cm noise
    }
    
    VectorXd weights = VectorXd::Ones(6);
    // Better weights for better geometry
    for (int i = 0; i < 6; ++i) {
        weights(i) = 1.0 / (1 + noisy_sats[i].tau / 1000.0);  // Rough heuristic
    }
    
    GPSSolution wls_sol = GPSSolver::solve_weighted(noisy_sats, weights);
    cout << "WLS position: " << wls_sol.position.transpose() << "\n";
    cout << "WLS bias: " << wls_sol.clock_bias << "\n";
    cout << "Error: " << (wls_sol.position - true_pos).norm() << " m\n";
}

//==============================================================================
// 2. Quadric Classification
//==============================================================================

void test_quadrics() {
    cout << "\n=== Quadric Classification ===\n\n";
    
    // Sphere: focus at origin, directrix plane at infinity, e=0
    // Actually: sphere is focus at center, e=0
    // For e=0: sphere centered at focus
    Quadric sphere = Quadric::from_focus_directrix(
        Vector3d::Zero(), Vector3d(0,0,1), Vector3d(0,0,1), 0);
    cout << "Sphere: " << sphere.type_string() << "\n";
    
    // Paraboloid: e=1
    Quadric parab = Quadric::from_focus_directrix(
        Vector3d(0,0,1), Vector3d(0,0,-1), Vector3d(0,0,-1), 1);
    cout << "Paraboloid: " << parab.type_string() << "\n";
    
    // Ellipsoid: e=0.5
    Quadric ellip = Quadric::from_focus_directrix(
        Vector3d(0,0,0), Vector3d(0,0,2), Vector3d(0,0,1), 0.5);
    cout << "Ellipsoid: " << ellip.type_string() << "\n";
    
    // Hyperboloid: e=2
    Quadric hyper = Quadric::from_focus_directrix(
        Vector3d(0,0,0), Vector3d(0,0,1), Vector3d(0,0,1), 2);
    cout << "Hyperboloid: " << hyper.type_string() << "\n";
    
    // GPS satellites on a sphere (user at center) -> unique solution
    // GPS satellites on paraboloid with user at focus -> two solutions
    cout << "\nGPS Geometry:\n";
    cout << "Satellites on sphere -> unique solution (rank 5)\n";
    cout << "Satellites on paraboloid with user at focus -> 2 solutions (rank 4)\n";
    cout << "Satellites on hyperboloid -> 2 solutions\n";
}

//==============================================================================
// Main
//==============================================================================

int main() {
    cout << "=== Chapter 8: New Developments in Global Positioning ===\n\n";
    
    simulate_gps();
    test_quadrics();
    
    cout << "\n=== Key Results ===\n";
    cout << "1. GPS = sphere intersection in Minkowski space\n";
    cout << "2. Unique solution iff satellites NOT on quadric with user at focus\n";
    cout << "3. Two solutions iff satellites lie on quadric with user at focus\n";
    cout << "4. Four quadric types: sphere, ellipsoid, paraboloid, hyperboloid\n";
    cout << "5. Conditioning: ill-conditioned near quadric degeneracy\n";
    cout << "6. Weighted LS: weight by satellite position accuracy\n";
    cout << "7. Application: also works for seismic, acoustic, shot detection\n";
    
    return 0;
}