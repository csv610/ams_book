//==============================================================================
// Chapter 5: Equivariant Neural Networks
// C++ Implementation of SO(2)/SO(3) Equivariant Convolutions
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include <random>
#include <cmath>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. Group Representations
//==============================================================================

// SO(2) representation: rotation by angle theta
// Acts on complex numbers: z -> e^{i*l*theta} * z
// For l=0: scalars (invariant)
// For l=1: vectors (equivariant)
struct SO2Representation {
    int max_freq;  // Maximum frequency L
    
    SO2Representation(int L) : max_freq(L) {}
    
    // Dimension of representation up to frequency L
    int dim() const { return 1 + 2 * max_freq; }  // l=0 + 2*L for l>=1
    
    // Rotation matrix for frequency l
    Matrix2d rotation_matrix(double theta, int l) const {
        double c = cos(l * theta);
        double s = sin(l * theta);
        Matrix2d R;
        R << c, -s,
             s,  c;
        return R;
    }
    
    // Full block-diagonal representation matrix
    MatrixXd rep_matrix(double theta) const {
        MatrixXd R = MatrixXd::Zero(dim(), dim());
        int idx = 0;
        // l=0: scalar (1x1 block)
        R(idx, idx) = 1.0;
        idx = 1;
        // l=1..max_freq: 2x2 blocks
        for (int l = 1; l <= max_freq; ++l) {
            R.block(idx, idx, 2, 2) = rotation_matrix(theta, l);
            idx += 2;
        }
        return R;
    }
};

// SO(3) representation using Wigner D-matrices
// For practical implementation, use scalar + vector (l=0,1) fields
struct SO3Representation {
    // Scalar field (l=0): invariant
    // Vector field (l=1): equivariant under 3x3 rotation matrix
    // Higher l: spherical tensors
    
    static Matrix3d rotation_matrix(const Vector3d& axis, double theta) {
        // Rodrigues formula
        Matrix3d K;
        K << 0, -axis(2), axis(1),
             axis(2), 0, -axis(0),
             -axis(1), axis(0), 0;
        return Matrix3d::Identity() + sin(theta) * K + (1-cos(theta)) * K * K;
    }
};

//==============================================================================
// 2. Equivariant Convolution Layers
//==============================================================================

// Steerable CNN for SO(2) - using complex-valued features
struct SO2Conv {
    int in_channels, out_channels, max_freq;
    int kernel_size;
    
    // Radial profiles: small MLPs for each frequency transition
    // l_in -> l_out with m = l_out - l_in
    vector<MatrixXd> radial_weights;  // Simplified: fixed radial basis
    
    SO2Conv(int C_in, int C_out, int L, int K) 
        : in_channels(C_in), out_channels(C_out), max_freq(L), kernel_size(K) {
        // Initialize radial profiles
        int n_freq_transitions = (L+1)*(L+1);  // All l_in, l_out
        radial_weights.resize(n_freq_transitions);
        for (auto& W : radial_weights) {
            W = MatrixXd::Random(C_out, C_in) * 0.1;
        }
    }
    
    // Simplified forward (in practice uses FFT)
    // Input: [batch, C_in, L+1, 2, H, W] complex features
    // Output: [batch, C_out, L+1, 2, H, W]
    void forward(const vector<MatrixXcd>& input, vector<MatrixXcd>& output) {
        // Placeholder - real implementation uses steerable filters
        // with Clebsch-Gordan coefficients for frequency mixing
        output = input;  // Placeholder
    }
};

// Simplified: Scalar + Vector fields for SO(3) equivariance
struct SO3Conv {
    int in_scalar, in_vector, out_scalar, out_vector;
    
    // Weight matrices
    MatrixXd W_ss, W_sv, W_vs, W_vv;
    
    SO3Conv(int C_s_in, int C_v_in, int C_s_out, int C_v_out)
        : in_scalar(C_s_in), in_vector(C_v_in), out_scalar(C_s_out), out_vector(C_v_out) {
        W_ss = MatrixXd::Random(C_s_out, C_s_in) * 0.1;
        W_sv = MatrixXd::Random(C_v_out * 3, C_s_in) * 0.1;
        W_vs = MatrixXd::Random(C_s_out, C_v_in * 3) * 0.1;
        W_vv = MatrixXd::Random(C_v_out * 3, C_v_in * 3) * 0.1;
    }
    
    // Forward pass: scalars and vectors transform differently under rotation
    // Scalars: standard conv
    // Vectors: conv + cross product terms
    pair<MatrixXd, MatrixXd> forward(const MatrixXd& scalars, const MatrixXd& vectors) {
        // scalars: [N, C_s_in], vectors: [N, C_v_in * 3]
        MatrixXd out_s = W_ss * scalars + W_vs * vectors;
        MatrixXd out_v = W_sv * scalars + W_vv * vectors;
        return {out_s, out_v};
    }
    
    // Apply rotation to features
    pair<MatrixXd, MatrixXd> rotate(const MatrixXd& scalars, const MatrixXd& vectors, 
                                     const Matrix3d& R) {
        MatrixXd s = scalars;  // Scalars invariant
        int N = vectors.rows() / 3;
        MatrixXd v = MatrixXd::Zero(vectors.rows(), vectors.cols());
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < vectors.cols(); ++c) {
                Vector3d v_vec = vectors.block(n*3, c, 3, 1);
                v.block(n*3, c, 3, 1) = R * v_vec;
            }
        }
        return {s, v};
    }
    
    // Check equivariance
    void test_equivariance() {
        Matrix3d R = SO3Representation::rotation_matrix(Vector3d::UnitZ(), M_PI/4);
        
        int N = 10, C_s = in_scalar, C_v = in_vector;
        MatrixXd s = MatrixXd::Random(N, C_s);
        MatrixXd v = MatrixXd::Random(N, C_v * 3);
        
        // f(R*x)
        auto [s_rot, v_rot] = rotate(s, v, R);
        auto [f_s_rot, f_v_rot] = forward(s_rot, v_rot);
        
        // R*f(x)
        auto [f_s, f_v] = forward(s, v);
        auto [_, f_v_rot2] = rotate(f_s, f_v, R);
        
        double err = (f_v_rot - f_v_rot2).norm() / f_v_rot.norm();
        cout << "Equivariance error: " << err << (err < 1e-10 ? " (PASS)" : " (FAIL)") << "\n";
    }
};

//==============================================================================
// 3. Reynolds Operator (Projection onto Equivariant Subspace)
//==============================================================================

// For finite group G, Reynolds operator: R(M) = (1/|G|) sum_{g in G} rho_out(g) M rho_in(g^{-1})
MatrixXd reynolds_operator(const MatrixXd& M, 
                           const vector<MatrixXd>& rho_in,
                           const vector<MatrixXd>& rho_out) {
    int d_in = rho_in[0].cols();
    int d_out = rho_out[0].rows();
    MatrixXd result = MatrixXd::Zero(d_out, d_in);
    
    for (size_t i = 0; i < rho_in.size(); ++i) {
        result += rho_out[i] * M * rho_in[i].transpose();
    }
    return result / rho_in.size();
}

// Example: D4 (dihedral group of square) on 2x2 images
vector<MatrixXd> D4_representation_2x2() {
    // 8 elements: 4 rotations + 4 reflections
    vector<MatrixXd> reps;
    
    // Rotation by 90 degrees
    MatrixXd R90(4,4); R90.setZero();
    R90(0,1) = R90(1,3) = R90(3,2) = R90(2,0) = 1;
    reps.push_back(MatrixXd::Identity(4,4));  // e
    reps.push_back(R90);                       // r
    reps.push_back(R90*R90);                   // r^2
    reps.push_back(R90*R90*R90);               // r^3
    
    // Reflections
    MatrixXd H(4,4); H.setZero(); H(0,1)=H(1,0)=H(2,3)=H(3,2)=1;  // horizontal
    MatrixXd V(4,4); V.setZero(); V(0,2)=V(2,0)=V(1,3)=V(3,1)=1;  // vertical
    MatrixXd D1(4,4); D1.setZero(); D1(0,0)=D1(1,3)=D1(3,1)=D1(2,2)=1;  // diag
    MatrixXd D2(4,4); D2.setZero(); D2(0,3)=D2(3,0)=D2(1,1)=D2(2,2)=1;  // anti-diag
    
    reps.push_back(H); reps.push_back(V); reps.push_back(D1); reps.push_back(D2);
    return reps;
}

//==============================================================================
// 4. Universal Approximation Test
//==============================================================================

void test_universal_approximation() {
    cout << "\n=== Equivariant Universal Approximation ===\n";
    cout << "Theorem: Equivariant networks can approximate any continuous\n";
    cout << "equivariant function if nonlinearities separate orbits.\n\n";
    
    // For permutation group S_n, ReLU works
    // For SO(3), need gated nonlinearities
    cout << "S_n: ReLU is equivariant (permutes coordinates)\n";
    cout << "SO(3): Need gated nonlinearities sigma(||x||) * x/||x||\n";
    cout << "Key: Nonlinearity must NOT be invariant\n";
}

//==============================================================================
// Main
//==============================================================================
int main() {
    cout << "=== Chapter 5: Equivariant Neural Networks ===\n\n";
    
    // SO(2) representation
    cout << "=== SO(2) Representation ===\n";
    SO2Representation so2(3);
    cout << "Max freq L=3, dim = " << so2.dim() << "\n";
    
    double theta = M_PI/4;
    MatrixXd R = so2.rep_matrix(theta);
    cout << "Rotation by 45° matrix (block diag):\n";
    for (int i = 0; i < R.rows(); ++i) {
        for (int j = 0; j < R.cols(); ++j) {
            if (abs(R(i,j)) > 1e-10) cout << "R(" << i << "," << j << ") = " << R(i,j) << "\n";
        }
    }
    
    // SO(3) convolution
    cout << "\n=== SO(3) Convolution ===\n";
    SO3Conv so3conv(32, 16, 32, 16);
    so3conv.test_equivariance();
    
    // Reynolds operator
    cout << "\n=== Reynolds Operator (D4 on 2x2) ===\n";
    auto D4 = D4_representation_2x2();
    MatrixXd M = MatrixXd::Random(4, 4);
    MatrixXd M_eq = reynolds_operator(M, D4, D4);
    cout << "Original M:\n" << M << "\n";
    cout << "Equivariant projection:\n" << M_eq << "\n";
    
    // Check D4 equivariance
    for (const auto& g : D4) {
        MatrixXd lhs = g * M_eq * g.transpose();
        double err = (lhs - M_eq).norm();
        cout << "Equivariance error: " << err << (err < 1e-10 ? " PASS" : " FAIL") << "\n";
    }
    
    // Universal approximation
    test_universal_approximation();
    
    // Applications
    cout << "\n=== Applications ===\n";
    cout << "1. Molecular property prediction (AlphaFold, NequiP, MACE)\n";
    cout << "2. Point cloud processing (PointNet++ with SO(3) equivariance)\n";
    cout << "3. Medical imaging (rotation-equivariant segmentation)\n";
    cout << "4. Physics simulation (Hamiltonian dynamics with symmetries)\n";
    cout << "5. Cosmology (spherical CNNs for CMB analysis)\n";
    
    return 0;
}