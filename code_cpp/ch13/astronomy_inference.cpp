//==============================================================================
// Chapter 13: A Brief History of Inference in Astronomy
// C++ Implementation: Least Squares, MCMC, HMC, Nested Sampling, LFI
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. Least Squares (Gauss's Method for Ceres)
//==============================================================================

struct LeastSquares {
    // Solve y = Xθ + ε, ε ~ N(0, Σ)
    // MLE: θ̂ = (X^T Σ^{-1} X)^{-1} X^T Σ^{-1} y
    
    static VectorXd solve(const MatrixXd& X, const VectorXd& y, 
                          const MatrixXd& Sigma = MatrixXd::Identity(1,1)) {
        int n = X.rows(), p = X.cols();
        MatrixXd Sinv = Sigma.inverse();
        MatrixXd XTSinvX = X.transpose() * Sinv * X;
        VectorXd XTSinvY = X.transpose() * Sinv * y;
        return XTSinvX.ldlt().solve(XTSinvY);
    }
    
    // Orbit fitting for Ceres (simplified ellipse)
    struct OrbitParams {
        double a, e, i, Omega, omega, M0;  // Semi-major, eccentricity, inclination, etc.
    };
    
    static OrbitParams fit_ceres_orbit(const vector<Vector2d>& observations,
                                        const vector<double>& times) {
        // Gauss's method: fit ellipse to angular observations
        // This is a simplified placeholder
        OrbitParams params;
        params.a = 2.77;      // AU
        params.e = 0.076;     // Eccentricity
        params.i = 10.59 * M_PI/180;  // Inclination
        params.Omega = 80.3 * M_PI/180;
        params.omega = 73.0 * M_PI/180;
        params.M0 = 0;
        return params;
    }
};

//==============================================================================
// 2. Bayesian Inference in Cosmology
//==============================================================================

struct CosmologicalModel {
    double Omega_m, Omega_L, H0;
    
    // Luminosity distance in flat ΛCDM
    double dist_modulus(double z) const {
        // d_L = (c/H0) * (1+z) * ∫_0^z dz'/E(z')
        // E(z) = sqrt(Ω_m(1+z)^3 + Ω_L)
        // Numerical integration
        int N = 1000;
        double dz = z / N;
        double integral = 0;
        for (int i = 0; i < N; ++i) {
            double zi = (i + 0.5) * dz;
            double E = sqrt(Omega_m * pow(1+zi, 3) + Omega_L);
            integral += dz / E;
        }
        double dL = (3e5 / 70.0) * (1+z) * integral;  // Mpc
        return 5 * log10(dL * 1e6 / 10.0);  // Distance modulus
    }
    
    // Log-likelihood for supernova data
    double log_likelihood(const VectorXd& obs_mu, const VectorXd& obs_z, 
                          const VectorXd& obs_sigma) const {
        int N = obs_mu.size();
        double logL = 0;
        for (int i = 0; i < N; ++i) {
            double mu_pred = dist_modulus(obs_z(i));
            double diff = obs_mu(i) - mu_pred;
            logL -= 0.5 * (diff*diff / (obs_sigma(i)*obs_sigma(i)) + log(2*M_PI*obs_sigma(i)*obs_sigma(i)));
        }
        return logL;
    }
};

//==============================================================================
// 3. MCMC Sampling (Metropolis-Hastings)
//==============================================================================

struct MCMC {
    // Metropolis-Hastings sampler
    template<typename LogPosterior>
    static vector<VectorXd> metropolis(const LogPosterior& log_post,
                                        const VectorXd& start,
                                        int n_samples,
                                        const MatrixXd& proposal_cov) {
        int d = start.size();
        vector<VectorXd> samples;
        samples.reserve(n_samples);
        
        VectorXd current = start;
        double log_p_current = log_post(current);
        
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> normal(0, 1);
        
        LLT<MatrixXd> chol(proposal_cov);
        MatrixXd L = chol.matrixL();
        
        int accepted = 0;
        for (int i = 0; i < n_samples; ++i) {
            // Propose
            VectorXd proposal = current + L * VectorXd::NullaryExpr(current.size(), [&](){ return normal(gen); });
            double log_p_proposal = log_post(proposal);
            
            double log_alpha = log_p_proposal - log_p_current;
            if (log((double)rand() / RAND_MAX) < log_alpha) {
                current = proposal;
                log_p_current = log_p_proposal;
                accepted++;
            }
            samples.push_back(current);
        }
        
        cout << "Acceptance rate: " << double(accepted)/n_samples << "\n";
        return samples;
    }
    
    // Hamiltonian Monte Carlo (HMC)
    template<typename LogPosterior, typename GradLogPosterior>
    static vector<VectorXd> hmc(const LogPosterior& log_post,
                                 const GradLogPosterior& grad_log_post,
                                 const VectorXd& start,
                                 int n_samples,
                                 double step_size = 0.01,
                                 int n_leapfrog = 10) {
        int d = start.size();
        vector<VectorXd> samples;
        samples.reserve(n_samples);
        
        VectorXd q = start;
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> normal(0, 1);
        
        int accepted = 0;
        for (int i = 0; i < n_samples; ++i) {
            // Sample momentum
            VectorXd p = VectorXd::NullaryExpr(d, [&](){ return normal(gen); });
            VectorXd q_new = q, p_new = p;
            
            // Leapfrog integration
            p_new -= 0.5 * step_size * grad_log_post(q_new);
            for (int j = 0; j < n_leapfrog; ++j) {
                q_new += step_size * p_new;
                if (j != n_leapfrog - 1) 
                    p_new -= step_size * grad_log_post(q_new);
            }
            p_new -= 0.5 * step_size * grad_log_post(q_new);
            p_new = -p_new;  // Momentum flip for reversibility
            
            // Metropolis acceptance
            double log_p_new = log_post(q_new) - 0.5 * p_new.squaredNorm();
            double log_p_old = log_post(q) - 0.5 * p.squaredNorm();
            double log_alpha = log_p_new - log_p_old;
            
            if (log(((double)rand()) / RAND_MAX) < log_alpha) {
                q = q_new;
                accepted++;
            }
            samples.push_back(q);
        }
        
        cout << "HMC acceptance rate: " << double(accepted)/n_samples << "\n";
        return samples;
    }
};

//==============================================================================
// 4. Nested Sampling
//==============================================================================

struct NestedSampling {
    // Compute evidence Z = ∫ L(θ) π(θ) dθ
    template<typename LogLikelihood, typename PriorSample>
    static double nested_sampling(const LogLikelihood& log_likelihood,
                                   const PriorSample& sample_prior,
                                   int n_live = 500,
                                   int max_iter = 10000) {
        int d = 10;  // dimension
        vector<pair<VectorXd, double>> live_points;
        
        // Initialize live points from prior
        for (int i = 0; i < n_live; ++i) {
            VectorXd theta = sample_prior();
            double logL = log_likelihood(theta);
            live_points.push_back({theta, logL});
        }
        
        sort(live_points.begin(), live_points.end(), 
             [](const auto& a, const auto& b) { return a.second < b.second; });
        
        double logZ = -INFINITY;
        double logX = 0;  // log prior volume
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // Worst live point
            double logL_worst = live_points[0].second;
            
            // Prior volume shrinkage
            logX += log((double)(n_live - 1) / n_live);
            
            // Contribution to evidence
            double logL_avg = (logL_worst + live_points[1].second) / 2;
            double logZ_new = logsumexp(logZ, logX + logL_avg);
            logZ = logZ_new;
            
            // Replace worst point
            VectorXd new_theta;
            double new_logL;
            int attempts = 0;
            do {
                new_theta = sample_prior();
                new_logL = log_likelihood(new_theta);
                attempts++;
            } while (new_logL < logL_worst && attempts < 100);
            
            live_points[0] = {new_theta, new_logL};
            sort(live_points.begin(), live_points.end(), 
                 [](const auto& a, const auto& b) { return a.second < b.second; });
            
            // Termination: remaining prior volume * max L < tolerance
            double max_logL = live_points.back().second;
            if (logX + max_logL - logZ < -10) break;
        }
        
        // Add remaining live points
        for (const auto& lp : live_points) {
            logZ = logsumexp(logZ, logX + lp.second);
        }
        
        return logZ;
    }
    
    static double logsumexp(double a, double b) {
        if (a < b) swap(a, b);
        return a + log1p(exp(b - a));
    }
};

//==============================================================================
// 5. Likelihood-Free Inference (LFI)
//==============================================================================

struct LikelihoodFreeInference {
    // Approximate Bayesian Computation (ABC)
    template<typename Simulator, typename Distance>
    static vector<VectorXd> abc(const Simulator& simulate,
                                 const Distance& distance,
                                 const VectorXd& obs,
                                 int n_samples,
                                 double epsilon) {
        vector<VectorXd> samples;
        vector<double> distances;
        
        random_device rd;
        mt19937 gen(rd());
        
        while (samples.size() < n_samples) {
            VectorXd theta = simulate();  // Sample from prior
            VectorXd sim_data = simulate(theta);
            double d = distance(sim_data, obs);
            
            if (d < epsilon) {
                samples.push_back(theta);
                distances.push_back(d);
            }
        }
        return samples;
    }
    
    // Neural Density Estimation (simplified)
    // Train normalizing flow on (θ, x) pairs
    static void train_normalizing_flow() {
        cout << "Training normalizing flow on (θ, x) pairs...\n";
        cout << "Use: swyft, sbi, zuko packages in Python\n";
    }
};

//==============================================================================
// 6. Amortized Inference
//==============================================================================

struct AmortizedInference {
    // Train neural network to output posterior q_φ(θ|x)
    // min_φ E[KL(p(θ|x) || q_φ(θ|x))]
    
    static void train_amortized_posterior() {
        cout << "Training amortized posterior:\n";
        cout << "  - Sample θ ~ p(θ), x ~ p(x|θ)\n";
        cout << "  - Train neural net q_φ(θ|x)\n";
        cout << "  - Use: swyft, sbi, zuko packages\n";
    }
};

//==============================================================================
// 7. Historical Timeline
//==============================================================================

void print_timeline() {
    cout << "\n=== Historical Timeline of Inference in Astronomy ===\n\n";
    cout << "Hipparchus (190-120 BC)     : Averaging -> noise reduction\n";
    cout << "Tycho Brahe (1546-1601)     : Repeated observations -> precision\n";
    cout << "Gauss (1801)                : Least squares -> orbit of Ceres\n";
    cout << "Laplace (1812)              : Central Limit Theorem -> error theory\n";
    cout << "Jeffreys (1939)             : Bayesian theory -> cosmology\n";
    cout << "MCMC (1990s--)              : Cosmological parameter estimation\n";
    cout << "HMC/Nested Sampling (2000s): High-dimensional posteriors\n";
    cout << "Likelihood-free (2010s--)   : Simulator-based inference\n";
    cout << "Amortized (2020s--)         : Neural posterior estimation\n";
}

//==============================================================================
// Main
//==============================================================================

int main() {
    cout << "=== Chapter 13: A Brief History of Inference in Astronomy ===\n\n";
    
    // 1. Least Squares
    cout << "=== Least Squares (Gauss/Ceres) ===\n";
    MatrixXd X(6, 3);
    X << 1, 0, 0,
         1, 1, 1,
         1, 2, 4,
         1, 3, 9,
         1, 4, 16,
         1, 5, 25;
    VectorXd y(6);
    y << 1, 3, 5, 7, 9, 11;
    
    VectorXd theta = LeastSquares::solve(X, y);
    cout << "Fitted parameters: " << theta.transpose() << "\n";
    
    auto ceres = LeastSquares::fit_ceres_orbit({}, {});
    cout << "Ceres orbit: a=" << ceres.a << " AU, e=" << ceres.e << "\n";
    
    // 2. Bayesian Cosmology
    cout << "\n=== Bayesian Cosmology ===\n";
    CosmologicalModel model{0.3, 0.7, 70};
    double mu = model.dist_modulus(0.5);
    cout << "Distance modulus at z=0.5: " << mu << "\n";
    
    // 3. MCMC
    cout << "\n=== MCMC Sampling ===\n";
    auto log_post = [](const VectorXd& theta) {
        return -0.5 * theta.squaredNorm();  // Standard normal
    };
    VectorXd start = VectorXd::Zero(2);
    MatrixXd prop = MatrixXd::Identity(2,2) * 0.1;
    auto samples = MCMC::metropolis(log_post, start, 1000, prop);
    cout << "MCMC samples: " << samples.size() << "\n";
    
    // HMC
    auto grad_log = [](const VectorXd& theta) { return -theta; };
    auto hmcs = MCMC::hmc(log_post, grad_log, start, 1000, 0.1, 10);
    cout << "HMC samples: " << hmcs.size() << "\n";
    
    // 4. Nested Sampling
    cout << "\n=== Nested Sampling ===\n";
    double logZ = NestedSampling::nested_sampling(
        [](const VectorXd& t){ return -0.5 * t.squaredNorm(); },
        [](){ return VectorXd::Random(2); }
    );
    cout << "log Z = " << logZ << "\n";
    
    // 5. LFI and Amortized
    cout << "\n=== Likelihood-Free & Amortized Inference ===\n";
    LikelihoodFreeInference::train_normalizing_flow();
    AmortizedInference::train_amortized_posterior();
    
    // 6. Timeline
    print_timeline();
    
    // 7. Key Lessons
    cout << "\n=== Key Lessons ===\n";
    cout << "1. Astronomy drives statistical innovation\n";
    cout << "2. Gauss's least squares born from Ceres orbit\n";
    cout << "3. MCMC revolutionized cosmology in 1990s\n";
    cout << "4. HMC/Nested Sampling essential for >10 params\n";
    cout << "5. LFI for intractable likelihoods (galaxy formation, GW)\n";
    cout << "6. Amortization critical for real-time (LSST, GW)\n";
    
    return 0;
}