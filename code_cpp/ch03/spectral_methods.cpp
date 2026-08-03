//==============================================================================
// Chapter 3: Spectral Methods in Microeconomics
// C++ Implementation: Spectral Analysis of Networks and Economic Models
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. Perron-Frobenius Theory for Nonnegative Matrices
//==============================================================================

struct PerronFrobeniusResult {
    double spectral_radius;      // ρ(A)
    VectorXd right_eigenvector;  // A * u = ρ u
    VectorXd left_eigenvector;   // v^T A = ρ v^T
    bool converged;
    int iterations;
};

PerronFrobeniusResult power_method(const MatrixXd& A, int max_iter = 1000, double tol = 1e-12) {
    int n = A.rows();
    VectorXd u = VectorXd::Ones(n) / sqrt(n);  // Start with uniform
    VectorXd v = VectorXd::Ones(n) / sqrt(n);
    
    double rho_old = 0;
    PerronFrobeniusResult result;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Right eigenvector
        u = A * u;
        double norm_u = u.norm();
        u /= norm_u;
        
        // Left eigenvector
        v = A.transpose() * v;
        double norm_v = v.norm();
        v /= norm_v;
        
        double rho = (v.transpose() * A * u).value() / (v.dot(u));
        
        if (iter > 0 && std::abs(rho - rho_old) < tol) {
            result.spectral_radius = rho;
            result.right_eigenvector = u;
            result.left_eigenvector = v;
            result.converged = true;
            result.iterations = iter;
            return result;
        }
        rho_old = rho;
    }
    
    result.converged = false;
    return result;
}

// Check if matrix is irreducible (strongly connected graph)
bool is_irreducible(const MatrixXd& A) {
    int n = A.rows();
    vector<vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A(i,j) > 1e-12) adj[i].push_back(j);
        }
    }
    
    // Kosaraju's algorithm for strong connectivity
    vector<bool> visited(n, false);
    vector<int> order;
    function<void(int)> dfs1 = [&](int u) {
        visited[u] = true;
        for (int v : adj[u]) if (!visited[v]) dfs1(v);
        order.push_back(u);
    };
    for (int i = 0; i < n; ++i) if (!visited[i]) dfs1(i);
    
    // Reverse graph
    vector<vector<int>> radj(n);
    for (int i = 0; i < n; ++i) 
        for (int j : adj[i]) radj[j].push_back(i);
    
    fill(visited.begin(), visited.end(), false);
    function<void(int)> dfs2 = [&](int u) {
        visited[u] = true;
        for (int v : radj[u]) if (!visited[v]) dfs2(v);
    };
    dfs2(order.back());
    
    return all_of(visited.begin(), visited.end(), [](bool v){ return v; });
}

//==============================================================================
// 2. DeGroot Learning Model
//==============================================================================

struct DeGrootModel {
    MatrixXd W;           // Weight matrix (row-stochastic)
    VectorXd opinions;    // Current opinions
    VectorXd stationary;  // π^T W = π^T
    
    DeGrootModel(const MatrixXd& W_) : W(W_) {
        int n = W.rows();
        opinions = VectorXd::Random(n).cwiseAbs();  // Random initial opinions
        opinions /= opinions.sum();
        compute_stationary();
    }
    
    void compute_stationary() {
        // π is left eigenvector of W for eigenvalue 1
        EigenSolver<MatrixXd> es(W.transpose());
        VectorXcd evals = es.eigenvalues();
        MatrixXcd evecs = es.eigenvectors();
        
        for (int i = 0; i < evals.size(); ++i) {
            if (std::abs(evals[i].real() - 1.0) < 1e-10) {
                stationary = evecs.col(i).real().cwiseAbs();
                stationary /= stationary.sum();
                break;
            }
        }
    }
    
    void step() {
        opinions = W * opinions;
    }
    
    void run(int steps) {
        for (int i = 0; i < steps; ++i) step();
    }
    
    // Check if network is wise sequence: max_i π_i^(n) -> 0 as n -> ∞
    static bool is_wise_sequence(const vector<MatrixXd>& Ws) {
        for (const auto& W : Ws) {
            DeGrootModel model(W);
            double max_pi = model.stationary.maxCoeff();
            if (max_pi > 0.1) return false;  // Arbitrary threshold
        }
        return true;
    }
};

//==============================================================================
// 3. Network Games (Ballester-Calvó-Armengol-Zenou)
//==============================================================================

struct NetworkGame {
    MatrixXd G;       // Adjacency matrix (weighted)
    VectorXd alpha;   // Standalone productivities
    double beta;      // Complementarity parameter
    VectorXd actions; // Equilibrium actions
    
    NetworkGame(const MatrixXd& G_, const VectorXd& alpha_, double beta_)
        : G(G_), alpha(alpha_), beta(beta_) {
        int n = G.rows();
        actions = VectorXd::Zero(n);
    }
    
    // Best response: a_i = α_i + β Σ_j G_ij a_j
    // Equilibrium: a* = (I - β G)^{-1} α
    VectorXd compute_equilibrium() {
        int n = G.rows();
        MatrixXd I = MatrixXd::Identity(n, n);
        MatrixXd M = I - beta * G;
        
        // Check spectral radius condition
        EigenSolver<MatrixXd> es(beta * G);
        VectorXcd evals = es.eigenvalues();
        double max_eval = 0;
        for (int i = 0; i < evals.size(); ++i) {
            max_eval = max(max_eval, std::abs(evals[i].real()));
        }
        
        if (max_eval >= 1.0) {
            cerr << "Warning: ρ(βG) = " << max_eval << " >= 1, no unique equilibrium\n";
            return VectorXd::Zero(G.rows());
        }
        
        actions = M.ldlt().solve(alpha);
        return actions;
    }
    
    // Katz-Bonacich centrality: b = (I - βG)^{-1} 1
    VectorXd katz_centrality() {
        int n = G.rows();
        MatrixXd I = MatrixXd::Identity(n, n);
        MatrixXd M = I - beta * G;
        return M.ldlt().solve(VectorXd::Ones(n));
    }
    
    // Equilibrium actions with heterogeneous alpha
    void solve() {
        actions = compute_equilibrium();
        // Action is proportional to Katz-Bonacich centrality when α_i constant
    }
};

//==============================================================================
// 4. Spectral Analysis of Networks
//==============================================================================

struct NetworkAnalysis {
    MatrixXd W;  // Row-stochastic weight matrix
    
    NetworkAnalysis(const MatrixXd& W_) : W(W_) {}
    
    // Eigenvector centrality (left Perron vector)
    VectorXd eigenvector_centrality() {
        EigenSolver<MatrixXd> es(W.transpose());
        VectorXcd evals = es.eigenvalues();
        MatrixXcd evecs = es.eigenvectors();
        
        for (int i = 0; i < evals.size(); ++i) {
            if (std::abs(evals[i].real() - 1.0) < 1e-10) {
                VectorXd centrality = evecs.col(i).real().cwiseAbs();
                return centrality / centrality.sum();
            }
        }
        return VectorXd::Zero(W.rows());
    }
    
    // Convergence rate in DeGroot model: 1 - |λ_2|
    double convergence_rate() {
        EigenSolver<MatrixXd> es(W);
        VectorXcd evals = es.eigenvalues();
        double lambda2 = 0;
        for (int i = 0; i < evals.size(); ++i) {
            double mag = std::abs(evals[i]);
            if (mag < 1.0 - 1e-10 && mag > lambda2) {
                lambda2 = mag;
            }
        }
        return 1.0 - lambda2;
    }
    
    // Wisdom of crowds test
    bool is_wise() {
        // Sequence of growing networks is wise iff max π_i^(n) -> 0
        // Here we check if centrality is diffuse
        VectorXd pi = eigenvector_centrality();
        return pi.maxCoeff() < 0.1;  // Heuristic threshold
    }
    
    // Prominent sequence detection
    bool has_prominent_sequence() {
        VectorXd pi = eigenvector_centrality();
        // Check if there's a subset with significant collective influence
        // Simplified: check if max entry is much larger than average
        return pi.maxCoeff() > 3 * pi.mean();
    }
};

//==============================================================================
// 5. Katz-Bonacich Centrality
//==============================================================================

VectorXd katz_bonacich(const MatrixXd& G, double beta) {
    int n = G.rows();
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd M = I - beta * G;
    return M.ldlt().solve(VectorXd::Ones(n));
}

//==============================================================================
// 6. Simulation Examples
//==============================================================================

void star_network_example() {
    cout << "=== Star Network (n=10) ===\n";
    int n = 10;
    MatrixXd W = MatrixXd::Zero(n, n);
    W(0, 0) = 0;
    for (int i = 1; i < n; ++i) {
        W(0, i) = 1.0 / (n - 1);
        W(i, 0) = 1.0;
    }
    
    NetworkAnalysis net(W);
    VectorXd pi = net.eigenvector_centrality();
    cout << "Central agent influence: " << pi(0) << "\n";
    cout << "Peripheral influence: " << pi(1) << "\n";
    cout << "Wise? " << (net.is_wise() ? "Yes" : "No") << "\n";
}

void line_network_example() {
    cout << "\n=== Line Network (n=10) ===\n";
    int n = 10;
    MatrixXd W = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        if (i > 0) W(i, i-1) = 0.5;
        if (i < n-1) W(i, i+1) = 0.5;
    }
    // Adjust endpoints
    W(0, 1) = 1.0;
    W(n-1, n-2) = 1.0;
    
    NetworkAnalysis net(W);
    VectorXd pi = net.eigenvector_centrality();
    cout << "Max influence: " << pi.maxCoeff() << "\n";
    cout << "Min influence: " << pi.minCoeff() << "\n";
    cout << "Wise? " << (net.is_wise() ? "Yes" : "No") << "\n";
}

void network_game_example() {
    cout << "\n=== Network Game (Ballester et al.) ===\n";
    int n = 5;
    MatrixXd G = MatrixXd::Zero(n, n);
    G << 0, 1, 0, 0, 0,
         1, 0, 1, 0, 0,
         0, 1, 0, 1, 0,
         0, 0, 1, 0, 1,
         0, 0, 0, 1, 0;
    
    VectorXd alpha = VectorXd::Constant(5, 1.0);
    double beta = 0.3;
    
    NetworkGame game(G, alpha, beta);
    VectorXd eq = game.compute_equilibrium();
    VectorXd centrality = game.katz_centrality();
    
    cout << "Equilibrium actions:\n" << eq.transpose() << "\n";
    cout << "Katz centrality:\n" << centrality.transpose() << "\n";
    cout << "Actions proportional to centrality? " 
         << (eq.normalized().isApprox(centrality.normalized()) ? "Yes" : "No") << "\n";
}

void degroot_example() {
    cout << "\n=== DeGroot Learning ===\n";
    int n = 10;
    MatrixXd W = MatrixXd::Random(n, n).cwiseAbs();
    W = W.rowwise().normalized();  // Row-stochastic
    
    DeGrootModel model(W);
    cout << "Initial opinions:\n" << model.opinions.transpose() << "\n";
    
    model.run(100);
    cout << "After 100 steps:\n" << model.opinions.transpose() << "\n";
    cout << "Consensus value: " << model.opinions.mean() << "\n";
    cout << "Stationary distribution π:\n" << model.stationary.transpose() << "\n";
    cout << "Social influence (π): max = " << model.stationary.maxCoeff() << "\n";
}

void erdos_renyi_wisdom() {
    cout << "\n=== Wisdom of Crowds in ER Graphs ===\n";
    for (int n : {50, 100, 200, 500}) {
        double p = log(n) / n;  // Connected regime
        MatrixXd W = MatrixXd::Zero(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && (rand() / double(RAND_MAX)) < p)
                    W(i, j) = 1.0;
        W = W.rowwise().normalized();
        
        NetworkAnalysis net(W);
        VectorXd pi = net.eigenvector_centrality();
        cout << "n=" << n << ", p=" << p << ", max_π=" << pi.maxCoeff() 
             << ", wise=" << (net.is_wise() ? "Yes" : "No") << "\n";
    }
}

int main() {
    cout << "=== Chapter 3: Spectral Methods in Microeconomics ===\n\n";
    
    // Perron-Frobenius
    cout << "=== Perron-Frobenius Theorem ===\n";
    MatrixXd A = MatrixXd::Random(5, 5).cwiseAbs();
    A.diagonal().setZero();
    for (int i = 0; i < 5; ++i) A(i, (i+1)%5) = 1.0;  // Make irreducible
    
    auto pf = power_method(A);
    cout << "ρ(A) = " << pf.spectral_radius << "\n";
    cout << "Right eigenvector (centrality): " << pf.right_eigenvector.transpose() << "\n";
    cout << "Left eigenvector (influence): " << pf.left_eigenvector.transpose() << "\n";
    cout << "Irreducible: " << (is_irreducible(A) ? "Yes" : "No") << "\n";
    
    // Economic applications
    star_network_example();
    line_network_example();
    network_game_example();
    degroot_example();
    erdos_renyi_wisdom();
    
    cout << "\n=== Key Results ===\n";
    cout << "1. Perron-Frobenius: ρ(A) has positive eigenvectors\n";
    cout << "2. DeGroot: opinions converge to π^T x(0) where π is left Perron vector\n";
    cout << "3. Network games: a* = (I - βG)^{-1} α, centrality = (I - βG)^{-1} 1\n";
    cout << "4. Wisdom of crowds iff max π_i -> 0 as n -> ∞\n";
    cout << "5. Network games: equilibrium actions proportional to Katz-Bonacich centrality\n";
    cout << "6. Spectral gap 1 - |λ_2| determines convergence speed\n";
    
    return 0;
}