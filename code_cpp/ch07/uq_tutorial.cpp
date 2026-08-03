//==============================================================================
// Chapter 7: Uncertainty Quantification Tutorial
// C++ Implementation of UQ methods: Entropy, KL, EnKF, Stochastic Surrogates
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. Information Measures
//==============================================================================

// Shannon entropy for discrete distribution
double shannon_entropy(const VectorXd& p) {
    double H = 0;
    for (int i = 0; i < p.size(); ++i) {
        if (p(i) > 0) H -= p(i) * log(p(i));
    }
    return H;
}

// Shannon entropy for Gaussian N(mu, Sigma)
double gaussian_entropy(const MatrixXd& Sigma) {
    int d = Sigma.rows();
    return 0.5 * d * (1 + log(2*M_PI)) + 0.5 * Sigma.llt().matrixL().diagonal().array().log().sum() * 2;
}

// KL divergence between two Gaussians
double kl_gaussian(const VectorXd& mu1, const MatrixXd& Sigma1,
                   const VectorXd& mu2, const MatrixXd& Sigma2) {
    int d = mu1.size();
    MatrixXd Sigma2_inv = Sigma2.inverse();
    double signal = 0.5 * (mu1 - mu2).transpose() * Sigma2_inv * (mu1 - mu2);
    double dispersion = 0.5 * (Sigma2_inv * Sigma1).trace() - 0.5 * d 
                        + 0.5 * (Sigma2.determinant() / Sigma1.determinant()).log();
    return signal + dispersion;
}

//==============================================================================
// 2. Dynamical Systems: Lorenz-63
//==============================================================================

struct Lorenz63 {
    double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
    
    Vector3d operator()(const Vector3d& x) const {
        return Vector3d(
            sigma * (x(1) - x(0)),
            x(0) * (rho - x(2)) - x(1),
            x(0) * x(1) - beta * x(2)
        );
    }
    
    Matrix3d jacobian(const Vector3d& x) const {
        Matrix3d J;
        J << -sigma, sigma, 0,
             rho - x(2), -1, -x(0),
             x(1), x(0), -beta;
        return J;
    }
};

// Integrate using RK4
Vector3d rk4_step(const Lorenz63& sys, const Vector3d& x, double dt) {
    Vector3d k1 = sys(x);
    Vector3d k2 = sys(x + 0.5*dt*k1);
    Vector3d k3 = sys(x + 0.5*dt*k2);
    Vector3d k4 = sys(x + dt*k3);
    return x + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4);
}

//==============================================================================
// 3. Ensemble Kalman Filter (EnKF)
//==============================================================================

struct EnKF {
    int n_state, n_obs, N_ensemble;
    MatrixXd H;      // Observation operator
    MatrixXd R;      // Observation noise covariance
    
    EnKF(int n_state_, int n_obs_, int N_, const MatrixXd& H_, const MatrixXd& R_)
        : n_state(n_state_), n_obs(n_obs_), N_ensemble(N_), H(H_), R(R_) {}
    
    // Analysis step: update ensemble with observation
    MatrixXd analysis_step(const MatrixXd& ensemble_f, const VectorXd& y) {
        // ensemble_f: n_state x N
        int N = ensemble_f.cols();
        
        // Ensemble mean and perturbations
        VectorXd mu_f = ensemble_f.rowwise().mean();
        MatrixXd X_f = ensemble_f.colwise() - mu_f;
        
        // Forecast in observation space
        MatrixXd Y_f = H * ensemble_f;
        VectorXd y_mean = Y_f.rowwise().mean();
        MatrixXd Y_pert = Y_f.colwise() - y_mean;
        
        // Perturbed observations
        MatrixXd Y_obs = MatrixXd::Zero(n_obs, N);
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0, 1);
        for (int i = 0; i < n_obs; ++i) {
            for (int j = 0; j < N; ++j) {
                Y_obs(i, j) = y(i) + sqrt(R(i,i)) * dist(gen);
            }
        }
        
        // Covariances
        MatrixXd Pf_xy = (X_f * Y_pert.transpose()) / (N - 1);
        MatrixXd Pf_yy = (Y_pert * Y_pert.transpose()) / (N - 1);
        
        // Kalman gain
        MatrixXd K = Pf_xy * (Pf_yy + R).inverse();
        
        // Analysis ensemble
        MatrixXd ensemble_a = ensemble_f + K * (Y_obs - Y_f);
        return ensemble_a;
    }
};

//==============================================================================
// 4. Stochastic Surrogate Modeling (Ornstein-Uhlenbeck)
//==============================================================================

struct StochasticSurrogate {
    // dx = -theta * x dt + sigma dW
    // Match equilibrium variance and decorrelation time
    
    double theta, sigma;
    double dt;
    
    StochasticSurrogate(double var_true, double tau_true, double dt_) 
        : dt(dt_) {
        // theta = 1/tau, sigma^2 = 2*theta*var
        theta = 1.0 / tau_true;
        sigma = sqrt(2 * theta * var_true);
    }
    
    // Exact OU step (no numerical integration error)
    double step(double x, mt19937& gen) {
        normal_distribution<> dist(0, 1);
        double dw = dist(gen) * sqrt(dt);
        return x * exp(-theta * dt) + sigma * sqrt((1 - exp(-2*theta*dt)) / (2*theta)) * dw;
    }
    
    // Calibrate from data
    static pair<double, double> calibrate(const VectorXd& data, double dt) {
        // MLE for OU parameters
        int n = data.size();
        double sum_x = data.sum();
        double sum_xx = data.squaredNorm();
        double sum_xy = 0, sum_yy = 0;
        for (int i = 1; i < n; ++i) {
            sum_xy += data(i-1) * data(i);
            sum_yy += data(i) * data(i);
        }
        
        double phi = sum_xy / (data.head(n-1).squaredNorm());
        double theta_est = -log(phi) / dt;
        double sigma2_est = (sum_yy - 2*phi*sum_xy + phi*phi*data.head(n-1).squaredNorm()) / (n-1);
        double sigma_est = sqrt(sigma2_est * 2 * theta_est / (1 - phi*phi));
        
        return {theta_est, sigma_est};
    }
};

//==============================================================================
// 5. Lorenz-63 Twin Experiment with EnKF
//==============================================================================

void lorenz_enkf_experiment() {
    cout << "\n=== Lorenz-63 EnKF Twin Experiment ===\n";
    
    Lorenz63 lorenz;
    double dt = 0.01;
    int n_steps = 10000;
    int obs_interval = 10;  // Observe every 10 steps (dt=0.1)
    
    // True trajectory
    Vector3d x_true(1, 1, 1);
    MatrixXd truth(3, n_steps/obs_interval + 1);
    truth.col(0) = x_true;
    
    for (int i = 1; i <= n_steps; ++i) {
        x_true = rk4_step(lorenz, x_true, dt);
        if (i % obs_interval == 0) {
            truth.col(i/obs_interval) = x_true;
        }
    }
    
    // Observations: observe x-component only with noise
    MatrixXd H(1, 3); H << 1, 0, 0;
    MatrixXd R(1, 1); R << 0.25;  // obs noise variance
    int N = 50;  // ensemble size
    
    EnKF enkf(3, 1, N, H, R);
    
    // Initialize ensemble
    MatrixXd ensemble(3, N);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0, 1);
    for (int i = 0; i < N; ++i) {
        ensemble.col(i) = truth.col(0) + Vector3d(dist(gen), dist(gen), dist(gen)) * 0.5;
    }
    
    // Assimilation
    MatrixXd rmse(N, truth.cols());
    for (int k = 1; k < truth.cols(); ++k) {
        // Forecast
        for (int i = 0; i < N; ++i) {
            Vector3d x = ensemble.col(i);
            for (int step = 0; step < obs_interval; ++step) {
                x = rk4_step(lorenz, x, dt);
            }
            ensemble.col(i) = x;
        }
        
        // Analysis
        VectorXd y = H * truth.col(k);  // Noisy obs (add noise)
        y(0) += 0.5 * dist(gen);
        ensemble = enkf.analysis_step(ensemble, y);
        
        // RMSE
        VectorXd mean = ensemble.rowwise().mean();
        rmse(k) = (mean - truth.col(k)).norm();
    }
    
    cout << "Final RMSE: " << rmse(truth.cols()-1) << "\n";
    cout << "Mean RMSE: " << rmse.rowwise().mean().mean() << "\n";
}

//==============================================================================
// Main
//==============================================================================
int main() {
    cout << "=== Chapter 7: Uncertainty Quantification Tutorial ===\n\n";
    
    // Entropy examples
    cout << "=== Shannon Entropy ===\n";
    VectorXd p1(3); p1 << 1, 0, 0;
    VectorXd p2(3); p2 << 1/3.0, 1/3.0, 1/3.0;
    VectorXd p3(3); p3 << 0.5, 0.3, 0.2;
    cout << "H([1,0,0]) = " << shannon_entropy(p1) << " (deterministic)\n";
    cout << "H([1/3,1/3,1/3]) = " << shannon_entropy(p2) << " (max)\n";
    cout << "H([0.5,0.3,0.2]) = " << shannon_entropy(p3) << "\n";
    
    // Gaussian entropy
    MatrixXd S1(2,2); S1 << 1, 0, 0, 1;
    MatrixXd S2(2,2); S2 << 2, 0.5, 0.5, 1;
    cout << "\nGaussian entropy:\n";
    cout << "N(0,I): " << gaussian_entropy(S1) << "\n";
    cout << "N(0,S): " << gaussian_entropy(S2) << "\n";
    
    // KL divergence
    VectorXd m1(2), m2(2); m1 << 0, 0; m2 << 0.5, 0;
    cout << "\nKL(N(0,I) || N([0.5,0], I)): " 
         << kl_gaussian(m1, S1, m2, S1) << "\n";
    
    // KL with different covariance
    cout << "KL(N(0,I) || N(0,2I)): " 
         << kl_gaussian(m1, S1, m1, 2*S1) << "\n";
    
    // Stochastic surrogate
    cout << "\n=== Stochastic Surrogate (OU Process) ===\n";
    StochasticSurrogate ss(2.0, 0.5, 0.01);  // var=2, tau=0.5
    mt19937 gen(42);
    double x = 0;
    for (int i = 0; i < 10; ++i) {
        x = ss.step(x, gen);
        cout << "Step " << i << ": x = " << x << "\n";
    }
    
    // Lorenz EnKF
    lorenz_enkf_experiment();
    
    // Summary
    cout << "\n=== Key UQ Concepts ===\n";
    cout << "1. Entropy measures spread; KL measures model inadequacy\n";
    cout << "2. Linear systems: uncertainty decays; Nonlinear: couples to mean\n";
    cout << "3. EnKF: Monte Carlo approximation of Kalman filter\n";
    cout << "4. Stochastic surrogates replace expensive PDEs with calibrated SDEs\n";
    cout << "5. Calibration: match equilibrium PDF and decorrelation time\n";
    cout << "5. Code: https://github.com/marandmath/UQ_tutorial_code\n";
    
    return 0;
}