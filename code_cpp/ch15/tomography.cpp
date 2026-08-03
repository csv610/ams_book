//==============================================================================
// Chapter 15: Deciphering Scrolls with Tomography
// C++ Implementation: CT Reconstruction Pipeline (FBP, Segmentation, MIP)
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <cmath>
#include <fstream>
#include <complex>
#include <fftw3.h>  // Requires FFTW library

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. CT Physics and Radon Transform
//==============================================================================

struct CTPhysics {
    // Parallel beam geometry
    // p(theta, s) = integral f(s cosθ - t sinθ, s sinθ + t cosθ) dt
    
    static VectorXd radon_transform(const MatrixXd& image, 
                                     const VectorXd& thetas,
                                     int n_detectors) {
        // Discrete radon transform (placeholder)
        int n_angles = thetas.size();
        VectorXd sinogram(n_angles * n_detectors);
        sinogram.setZero();
        // In practice, use projection-slice theorem or ray tracing
        return sinogram;
    }
    
    // Filtered Back-Projection (FBP)
    static MatrixXd fbp_reconstruction(const VectorXd& sinogram,
                                        const VectorXd& thetas,
                                        int n_detectors,
                                        int image_size,
                                        const string& filter = "ramp") {
        int n_angles = thetas.size();
        MatrixXd reconstruction = MatrixXd::Zero(image_size, image_size);
        
        // Filter in Fourier domain
        int n_fft = 1;
        while (n_fft < 2 * n_detectors) n_fft *= 2;
        
        // Filter: ramp |ω|, or windowed (Shepp-Logan, Hann, etc.)
        VectorXd filter_fft(n_fft);
        for (int k = 0; k < n_fft; ++k) {
            double freq = (k <= n_fft/2) ? k : k - n_fft;
            double omega = 2 * M_PI * freq / n_fft;
            if (filter == "ramp") {
                filter_fft(k) = abs(omega);
            } else if (filter == "shepp-logan") {
                filter_fft(k) = (omega == 0) ? 1.0 : abs(omega) * sin(omega/2) / (omega/2);
            } else if (filter == "hann") {
                filter_fft(k) = abs(omega) * 0.5 * (1 + cos(omega / (M_PI/2)));
            }
        }
        
        // For each angle: FFT -> multiply filter -> IFFT -> backproject
        // Placeholder: actual implementation uses FFTW
        
        cout << "FBP reconstruction: " << n_angles << " angles, " << n_detectors << " detectors\n";
        return reconstruction;
    }
};

//==============================================================================
// 2. Projection Acquisition (Simulated Arduino + Camera)
//==============================================================================

struct CTScanner {
    int n_projections = 800;
    int n_detectors = 512;
    double source_distance = 1000;  // mm
    double detector_distance = 1000; // mm
    
    struct Projection {
        VectorXd data;  // Detector values
        double angle;   // Radians
    };
    
    vector<Projection> acquire(const MatrixXd& phantom) {
        vector<Projection> projections;
        projections.reserve(n_projections);
        
        // Simulate Arduino stepper motor rotation
        for (int i = 0; i < n_projections; ++i) {
            double angle = 2 * M_PI * i / n_projections;
            
            // Simulate projector + camera
            Projection proj;
            proj.angle = angle;
            proj.data = VectorXd::Zero(512);
            
            // In practice: trigger camera via serial, capture frame
            // Here: simulate with radon transform
            
            projections.push_back(proj);
        }
        return projections;
    }
};

//==============================================================================
// 3. Preprocessing
//==============================================================================

struct Preprocessing {
    // Dark/flat field correction
    static VectorXd correct(const VectorXd& raw, 
                            const VectorXd& dark, 
                            const VectorXd& flat) {
        return (raw - dark).cwiseQuotient(flat - dark);
    }
    
    // Rotation axis determination
    static double find_rotation_axis(const MatrixXd& sinogram) {
        // Cross-correlation of 0° and 180° projections
        int n_angles = sinogram.rows();
        VectorXd proj0 = sinogram.row(0);
        VectorXd proj180 = sinogram.row(sinogram.rows()/2).reverse();
        
        // FFT cross-correlation
        int n = sinogram.cols();
        int n_fft = 1;
        while (n_fft < 2*n) n_fft *= 2;
        
        // Placeholder for FFT-based correlation
        // In practice: use FFTW
        
        // Simple peak detection
        VectorXd corr = (proj0.array() * proj180.array()).matrix();
        int n_peak;
        corr.maxCoeff(&n_peak);
        
        // Axis is at n/2 + offset
        return n/2.0 + (n_peak - n/2);
    }
    
    // Cropping to region of interest
    static MatrixXd crop(const MatrixXd& img, int x, int y, int w, int h) {
        return img.block(y, x, h, w);
    }
};

//==============================================================================
// 4. Segmentation (Spiral Layer Detection)
//==============================================================================

struct ScrollSegmentation {
    // Spiral model: r(θ) = a + b*θ
    struct SpiralParams {
        double cx, cy;  // Center
        double a, b;    // Spiral parameters
    };
    
    // Genetic algorithm for spiral fitting
    static SpiralParams fit_spiral(const MatrixXd& slice, 
                                    const SpiralParams& initial,
                                    int max_gen = 100) {
        SpiralParams best = initial;
        double best_fitness = 1e9;
        
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> mut(-0.1, 0.1);
        
        for (int gen = 0; gen < max_gen; ++gen) {
            // Population of spirals
            vector<SpiralParams> population(20);
            for (auto& s : population) {
                s = best;
                s.cx += mut(gen) * 0.5;
                s.cy += mut(gen) * 0.5;
                s.a += mut(gen);
                s.b += mut(gen);
            }
            
            // Evaluate fitness: correlation with image gradients
            for (const auto& s : population) {
                double fitness = evaluate_spiral(slice, s);
                if (fitness < best_fitness) {
                    best_fitness = fitness;
                    best = s;
                }
            }
        }
        return best;
    }
    
    static double evaluate_spiral(const MatrixXd& slice, const SpiralParams& s) {
        // Generate spiral mask and compare to image
        // Placeholder: compute sum of squared differences
        return 1.0;
    }
    
    // Cubic smoothing spline for final segmentation
    static vector<Vector2d> smooth_spiral(const SpiralParams& s, int n_points = 1000) {
        vector<Vector2d> points;
        for (int i = 0; i < n_points; ++i) {
            double theta = 4 * M_PI * i / n_points;
            double r = s.a + s.b * theta;
            points.push_back(Vector2d(s.cx + r*cos(theta), s.cy + r*sin(theta)));
        }
        return points;
    }
};

//==============================================================================
// 5. Maximum Intensity Projection (MIP)
//==============================================================================

struct MaximumIntensityProjection {
    // MIP(u,v) = max_z I_z(u,v)
    static MatrixXd mip(const vector<MatrixXd>& layers) {
        if (layers.empty()) return MatrixXd();
        
        int h = layers[0].rows();
        int w = layers[0].cols();
        MatrixXd result = MatrixXd::Constant(h, w, -INFINITY);
        
        for (const auto& layer : layers) {
            result = result.cwiseMax(layer);
        }
        return result;
    }
    
    // Invert if text is dark-on-light
    static MatrixXd invert_if_needed(const MatrixXd& img) {
        double mean = img.mean();
        if (mean < 0.5) {
            return 1.0 - img;  // Assuming normalized [0,1]
        }
        return img;
    }
};

//==============================================================================
// 6. Complete Pipeline
//==============================================================================

struct CTPipeline {
    CTScanner scanner;
    ScrollSegmentation segmenter;
    MaximumIntensityProjection mip;
    
    struct Result {
        MatrixXd reconstruction;
        vector<MatrixXd> layers;
        MatrixXd final_text;
    };
    
    Result run(const MatrixXd& phantom) {
        cout << "=== CT Pipeline ===\n";
        
        // 1. Acquire projections
        cout << "1. Acquiring projections...\n";
        auto projections = scanner.acquire(phantom);
        cout << "   Acquired " << projections.size() << " projections\n";
        
        // 2. Preprocessing
        cout << "2. Preprocessing...\n";
        double axis = Preprocessing::find_rotation_axis(MatrixXd::Zero(1,1));
        cout << "   Rotation axis: " << axis << "\n";
        
        // 3. Reconstruction (FBP)
        cout << "3. Filtered Back-Projection...\n";
        MatrixXd volume = MatrixXd::Zero(256, 256);  // Placeholder
        cout << "   Volume: 256 x 256 x 256\n";
        
        // 4. Segmentation
        cout << "4. Layer segmentation...\n";
        vector<MatrixXd> layers;
        // Spiral segmentation + flattening
        
        // 5. MIP
        cout << "5. Maximum Intensity Projection...\n";
        MatrixXd final_text = MaximumIntensityProjection::mip(layers);
        final_text = MaximumIntensityProjection::invert_if_needed(final_text);
        
        // 6. Export
        cout << "6. Exporting results...\n";
        // mesh.save_stl("scroll.stl");
        
        Result res;
        res.reconstruction = volume;
        res.layers = layers;
        res.final_text = final_text;
        return res;
    }
};

//==============================================================================
// 7. Main: Lorenz System Example
//==============================================================================

int main() {
    cout << "=== Chapter 15: Deciphering Scrolls with Tomography ===\n\n";
    
    // Create a simple phantom (text on spiral)
    MatrixXd phantom = MatrixXd::Zero(256, 256);
    // Add spiral text pattern
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) {
            double dx = i - 128;
            double dy = j - 128;
            double r = sqrt(dx*dx + dy*dy);
            double theta = atan2(dy, dx);
            double r_spiral = 20 + 2*theta;
            if (abs(r - r_spiral) < 2) {
                phantom(i, j) = 1.0;
            }
        }
    }
    
    // Run pipeline
    CTPipeline pipeline;
    auto result = pipeline.run(phantom);
    
    // Code Repository
    cout << "\n=== Code Repository ===\n";
    cout << "https://gitlab.com/csc1/archeolab/\n\n";
    
    // Further Reading
    cout << "\n=== Further Reading ===\n";
    cout << "Foschiatti et al. (2025) - Deciphering Scrolls with Tomography\n";
    cout << "Seales et al. (2016) - From Damage to Discovery via Virtual Unwrapping\n";
    cout << "Natterer (2001) - The Mathematics of Computerized Tomography\n";
    cout << "Feeman (2015) - The Mathematics of Medical Imaging\n";
    cout << "Vesuvius Challenge: https://vesuviuschallenge.org\n";
    
    return 0;
}