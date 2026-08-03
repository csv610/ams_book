# Chapter 1: Numerical Stability in Gaussian Elimination
# Complete Julia implementation with growth factor analysis

using LinearAlgebra, Random, Statistics, Printf
using SparseArrays, Plots

# ============================================================================
# 1. Basic Gaussian Elimination (no pivoting) - for educational purposes
# ============================================================================
function gaussian_elimination_no_pivot(A::Matrix{Float64})
    """Naive Gaussian elimination without pivoting - DO NOT USE IN PRACTICE"""
    n = size(A, 1)
    A = copy(A)
    L = Matrix{Float64}(I, n, n)
    
    for k = 1:n-1
        if abs(A[k,k]) < eps()
            error("Zero pivot encountered at step $k")
        end
        for i = k+1:n
            L[i,k] = A[i,k] / A[k,k]
            for j = k+1:n
                A[i,j] -= L[i,k] * A[k,j]
            end
        end
    end
    U = UpperTriangular(A)
    return L, U
end

# ============================================================================
# 2. Gaussian Elimination with Partial Pivoting (standard)
# ============================================================================
function gaussian_elimination_partial_pivot(A::Matrix{Float64})
    """Gaussian elimination with partial (row) pivoting"""
    n = size(A, 1)
    A = copy(A)
    L = Matrix{Float64}(I, n, n)
    P = Vector{Int}(1:n)
    
    for k = 1:n-1
        # Find pivot row
        i_max = argmax(abs.(A[k:n, k])) + k - 1
        if i_max != k
            # Swap rows in A
            A[k, :], A[i_max, :] = A[i_max, :], A[k, :]
            # Swap rows in L (only previous columns)
            L[k, 1:k-1], L[i_max, 1:k-1] = L[i_max, 1:k-1], L[k, 1:k-1]
            # Track permutation
            P[k], P[i_max] = P[i_max], P[k]
        end
        
        for i = k+1:n
            L[i,k] = A[i,k] / A[k,k]
            for j = k+1:n
                A[i,j] -= L[i,k] * A[k,j]
            end
        end
    end
    U = UpperTriangular(A)
    return L, U, P
end

function solve_lu(L, U, P, b)
    """Solve PA = LU system: A x = b"""
    n = length(b)
    # Apply permutation
    Pb = b[P]
    # Forward substitution: L y = Pb
    y = zeros(n)
    for i = 1:n
        y[i] = Pb[i] - dot(L[i, 1:i-1], y[1:i-1])
    end
    # Backward substitution: U x = y
    x = zeros(n)
    for i = n:-1:1
        x[i] = (y[i] - dot(U[i, i+1:n], x[i+1:n])) / U[i,i]
    end
    return x
end

# ============================================================================
# 3. Growth Factor Computation
# ============================================================================
function growth_factor(A::Matrix{Float64}; pivoting=:partial)
    """Compute the growth factor ρ = max|U_ij| / max|A_ij|"""
    n = size(A, 1)
    if pivoting == :none
        L, U = gaussian_elimination_no_pivot(A)
    elseif pivoting == :partial
        L, U, P = gaussian_elimination_partial_pivot(A)
    else
        error("Unknown pivoting strategy")
    end
    max_A = maximum(abs, A)
    max_U = maximum(abs, U.data)
    return max_U / max_A
end

# ============================================================================
# 4. Wilkinson Matrix (worst case for partial pivoting)
# ============================================================================
function wilkinson_matrix(n::Int)
    """Wilkinson matrix - worst case for partial pivoting"""
    A = Matrix{Float64}(I, n, n)
    for i = 1:n
        for j = i+1:n
            A[j,i] = -1.0
        end
    end
    A[:,end] .= 1.0
    return A
end

# ============================================================================
# 5. Benchmark: Exact vs Floating Point
# ============================================================================
function benchmark_exact_vs_float(n=1000)
    """Compare exact rational vs Float64 solve time and accuracy"""
    Random.seed!(42)
    A = rand(Bool, n, n)
    b = rand(Bool, n)
    
    # Float64 solve
    A_f = Float64.(A)
    b_f = Float64.(b)
    t1 = @elapsed x_f = A_f \ b_f
    
    # Exact rational solve (using Rational{BigInt})
    A_r = Rational{BigInt}.(A)
    b_r = Rational{BigInt}.(b)
    t2 = @elapsed x_r = A_r \ b_r
    
    # Relative error
    x_f_big = BigFloat.(x_f)
    x_r_float = Float64.(x_r)
    rel_error = norm(x_f_big - x_r_float) / norm(x_r_float)
    
    println("n = $n")
    println("  Float64 time: $(round(t1*1000, digits=2)) ms")
    println("  Exact time:   $(round(t2, digits=2)) s")
    println("  Speedup:      $(round(t2/t1))x")
    println("  Relative error: $rel_error")
    
    return t1, t2, rel_error
end

# ============================================================================
# 6. Growth Factor Experiments
# ============================================================================
function growth_factor_experiment()
    """Replicate the empirical growth factor table from the article"""
    Random.seed!(123)
    println("Growth factors for random matrices (partial pivoting):")
    println("n\tmedian\t99th percentile")
    
    for n in [100, 500, 1000, 5000]
        rhos = Float64[]
        for trial = 1:100
            A = randn(n, n)
            push!(rhos, growth_factor(A, pivoting=:partial))
        end
        println("$n\t$(round(median(rhos), digits=1))\t$(round(quantile(rhos, 0.99), digits=1))")
    end
    
    # Wilkinson matrix
    println("\nWilkinson matrix (worst case):")
    for n in [20, 50, 100]
        A = wilkinson_matrix(n)
        rho = growth_factor(A, pivoting=:partial)
        println("n=$n: ρ = $rho (theoretical max: $(2^(n-1)))")
    end
end

# ============================================================================
# 7. Condition Number and Forward Error Analysis
# ============================================================================
function error_analysis_demo(n=100)
    """Demonstrate relationship between condition number, growth factor, and error"""
    Random.seed!(42)
    
    # Well-conditioned matrix
    A_well = randn(n, n) + n*I
    κ_well = cond(A_well)
    
    # Ill-conditioned matrix
    A_ill = randn(n, n)
    A_ill = A_ill * A_ill'  # symmetric positive definite
    A_ill[1,1] = 1e-12  # Make nearly singular
    κ_ill = cond(A_ill)
    
    b = randn(n)
    x_true_well = A_well \ b
    x_true_ill = A_ill \ b
    
    # Solve with Float64 (which uses partial pivoting via LAPACK)
    x_f_well = Float64.(A_well) \ Float64.(b)
    x_f_ill = Float64.(A_ill) \ Float64.(b)
    
    err_well = norm(x_f_well - x_true_well) / norm(x_true_well)
    err_ill = norm(x_f_ill - x_true_ill) / norm(x_true_ill)
    
    println("Well-conditioned (κ ≈ $(round(κ_well, digits=1))):")
    println("  Relative error: $err_well")
    println("Ill-conditioned (κ ≈ $(round(κ_ill, digits=1))):")
    println("  Relative error: $err_ill")
    
    # Growth factors
    rho_well = growth_factor(A_well, pivoting=:partial)
    rho_ill = growth_factor(A_ill, pivoting=:partial)
    println("\nGrowth factors: well= $rho_well, ill= $rho_ill")
end

# ============================================================================
# Main
# ============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("=== Chapter 1: Gaussian Elimination Stability ===\n")
    
    # Run growth factor experiment
    growth_factor_experiment()
    
    println("\n=== Error Analysis ===")
    error_analysis_demo(100)
    
    println("\n=== Benchmark (smaller for demo) ===")
    benchmark_exact_vs_float(200)
end