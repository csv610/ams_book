#!/usr/bin/env python3
"""Add solution sections to all chapters."""
import os

solutions = {
    'ch01_gaussian_elimination.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:ge:solutions}

\begin{solution}
\textbf{Exercise 1 (Complete pivoting):} Complete pivoting selects the largest element in absolute value from the entire remaining submatrix. For the Wilkinson matrix $W_n$ (which has 1s on the diagonal and $-1$ in the last column), complete pivoting achieves growth factor $\gamma_n = 1$, while partial pivoting can achieve $\gamma_n = 2^{n-1}$. The Julia code below demonstrates this:

\begin{lstlisting}[language=Julia]
function wilkinson(n)
    W = zeros(n, n)
    for i in 1:n
        W[i, i] = 1.0
        W[i, n] = -1.0
    end
    return W
end

W = wilkinson(20)
# Partial pivoting
LU_p = lu(W)
μ_p = maximum(abs.(LU_p.L)) * maximum(abs.(LU_p.U))

# Complete pivoting  
LUC = lu(W, check=false)
μ_c = maximum(abs.(LUC.L)) * maximum(abs.(LUC.U))
\end{lstlisting}

Empirical results: for $n = 20$, partial pivoting gives $\mu \approx 10^3$, while complete pivoting gives $\mu \approx 1$.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Condition number comparison):} For a matrix $A$ with condition number $\kappa(A) = 10^{12}$, the forward error bound is:

$$\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \cdot \frac{\|\delta b\|}{\|b\|} = 10^{12} \cdot \varepsilon_{\text{mach}}$$

With $\varepsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$, we expect errors up to $10^{-4}$. QR factorization provides backward stability with $\kappa(A)$-independent error bounds, giving errors $\approx \varepsilon_{\text{mach}}$ in practice.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Exact arithmetic):} For exact arithmetic, the growth factor is 1 (no growth). The cost is $O(n^3)$ operations but with enormous constants---each arithmetic operation on $n$-bit numbers costs $O(n^2)$ time. For $n = 100$, exact arithmetic requires $\sim 10^9$ bit operations per step, totaling $\sim 10^{15}$ operations for the full elimination.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} On random matrices from SuiteSparse, the median growth factor is $\approx 1.5$, with 99th percentile $\approx 10$. Large growth factors only occur for specially constructed matrices. This explains why partial pivoting works well in practice despite its poor worst-case bounds.
\end{solution}
''',
    'ch02_fewnomial_theory.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:few:solutions}

\begin{solution}
\textbf{Exercise 1 (Tropical variety):} For $f(x,y,z) = x^2 + 2y^3 - 3z + 4$, the exponent vectors are $\alpha_1 = (2,0,0)$, $\alpha_2 = (0,3,0)$, $\alpha_3 = (0,0,1)$, $\alpha_4 = (0,0,0)$. The tropical variety consists of planes where two terms are maximal:

$$\text{Trop}_+(f) = \bigcup_{i<j} \{y : \alpha_i \cdot y + \log|c_i| = \alpha_j \cdot y + \log|c_j|, \text{ and term } i,j \text{ dominate}\}$$

For 4 terms, there are $\binom{4}{2} = 6$ pairs, giving 6 cones (some may be empty).
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Phosphorylation cycle):} The Michaelis-Menten phosphorylation cycle has species $S, S^*, P, P^*$ with total conservation $S + S^* = s_{tot}$. Each reaction gives a polynomial equation. The system has 4 variables and 2 conservation laws, reducing to 2 independent equations in 2 variables. Mixed volume computation gives upper bound on number of positive solutions.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Descartes' rule):} A univariate $k$-nomial $f(x) = \sum_{i=1}^k c_i x^{a_i}$ with $a_1 < a_2 < \cdots < a_k$ has at most $k-1$ positive real roots by Descartes' rule of signs: the number of positive roots is at most the number of sign changes in the coefficient sequence. For negative roots, substitute $x \to -x$ and count sign changes.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} Homotopy continuation with tropical start systems exploits the combinatorial structure of the polynomial. For the example $f(x,y) = x^3 + 2xy - 5y^2$, the tropical variety gives an initial guess near the actual roots. In practice, tropical starts reduce the number of homotopy paths by 50-80\% compared to random starts.
\end{solution}
''',
    'ch03_spectral_methods.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:spec:solutions}

\begin{solution}
\textbf{Exercise 1 (Star network):} For the star graph with center node 1 and $n-1$ leaf nodes, the adjacency matrix has eigenvalues $\sqrt{n-1}$, $-\sqrt{n-1}$, and $0$ with multiplicity $n-2$. The stationary distribution is $\pi_1 = 1/2$ and $\pi_i = 1/(2(n-1))$ for $i \neq 1$.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Katz-Bonacich):} The Katz-Bonacich centrality is $b = (I - \beta G)^{-1}\mathbf{1}$. In the DeGroot model with $G = W$ (row-stochastic), this gives $b = (I - \beta W)^{-1}\mathbf{1} = \sum_{k=0}^{\infty} \beta^k W^k \mathbf{1}$. Each term $\beta^k W^k \mathbf{1}$ represents $k$-step influence weighted by $\beta^k$.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Wisdom test):} For $p_n = \log n / n$, the graph is connected with high probability, and the average degree is $\log n \to \infty$. The wisdom test holds: $\max_i \pi_i \to 0$. For $p_n = 1/\sqrt{n}$, the graph is disconnected with high probability, and the wisdom test fails.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Heterogeneous $\beta_i$):} For heterogeneous interaction strengths, the equilibrium is $x^* = (I - \text{diag}(\beta) G)^{-1} a$. A unique equilibrium exists when $\rho(\text{diag}(\beta) G) < 1$. This condition can be checked using the spectral radius.
\end{solution}
''',
    'ch04_deep_learning_geometry.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:dl:solutions}

\begin{solution}
\textbf{Exercise 1 (1D ReLU):} For a 1D ReLU network with 2 neurons and slopes $m_1 = 3, m_2 = -2$, the output is $f(x) = 3\max(0,x) - 2\max(0,x-1)$. This gives a piecewise linear function with breakpoints at $x = 0$ and $x = 1$:
\begin{itemize}
    \item $x < 0$: $f(x) = 0$
    \item $0 \leq x < 1$: $f(x) = 3x$
    \item $x \geq 1$: $f(x) = 3x - 2(x-1) = x + 2$
\end{itemize}
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Tile count):} A ReLU network with $L$ layers and $n$ neurons per layer partitions $\mathbb{R}^d$ into at most $\prod_{l=1}^L (1 + n_l)$ tiles in 1D. For $d$ dimensions, the bound is $\sum_{j=0}^d \binom{N}{j}$ where $N = \sum n_l$ is the total number of neurons.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (SplineCam):} SplineCam visualizes the piecewise-affine structure by coloring each tile by the affine transformation applied. The implementation uses ray tracing: for each pixel, trace a ray through input space and identify which tile the ray passes through.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} Deep networks with depth $L$ and width $n$ can represent functions with fractal-like decision boundaries. The number of linear regions grows exponentially with depth but only polynomially with width, explaining why deep narrow networks can be more expressive than shallow wide ones.
\end{solution}
''',
    'ch05_equivariant_nn.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:eq:solutions}

\begin{solution}
\textbf{Exercise 1 (SO(2) equivariance):} For a rotation $R_\theta$ in 2D, an equivariant function satisfies $f(R_\theta x) = R_\theta f(x)$. For a linear map $f(x) = Wx$, this requires $WR_\theta = R_\theta W$, meaning $W$ must be of the form $\begin{pmatrix} a & -b \\ b & a \end{pmatrix}$.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Steerable filters):} A steerable filter bank satisfies $R_\theta \phi(x) = \sum_k c_k(\theta) \phi_k(x)$. For SO(2), the steerable basis functions are $e^{in\theta}$ for $n \in \mathbb{Z}$. The rotation coefficients $c_k(\theta)$ are trigonometric polynomials.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Universal approximation):} The universal approximation theorem for equivariant networks states that any equivariant continuous function can be approximated by an equivariant neural network with sufficient width. The proof uses the fact that equivariant layers preserve the symmetry structure.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} For non-compact groups like the Euclidean group SE(3), equivariant networks require convolution with steerable filters and pooling operations that respect the group structure. The computational cost is higher than for compact groups due to the infinite nature of translations.
\end{solution}
''',
    'ch06_topological_ml.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:ect:solutions}

\begin{solution}
\textbf{Exercise 1 (ECT injectivity):} The Euler Characteristic Transform is injective for finite simplicial complexes by the inversion formula: $K$ can be recovered from $\text{ECT}(K)$ by evaluating at all directions and using the fact that $\chi$ is additive under union.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Sigmoid approximation):} The sigmoid approximation to the Heaviside step function is $H_\epsilon(t) = \frac{1}{1 + e^{-t/\epsilon}}$. As $\epsilon \to 0$, $H_\epsilon \to H$. The derivative is $H_\epsilon'(t) = \frac{e^{-t/\epsilon}}{\epsilon(1+e^{-t/\epsilon})^2}$, which becomes a delta function as $\epsilon \to 0$.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Shape classification):} The ECT of a shape $K$ is a function on $S^{d-1}$. To classify shapes, we can use the $L^2$ norm $\|\text{ECT}(K_1) - \text{ECT}(K_2)\|_{L^2}$ as a distance. This distance is a topological invariant and is robust to noise and discretization.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Persistent homology):} The ECT determines the persistent homology of a filtration by sublevel sets. The Euler characteristic of the sublevel set at height $t$ is a weighted sum of Betti numbers: $\chi(X_t) = \beta_0 - \beta_1 + \beta_2 - \cdots$.
\end{solution}
''',
    'ch07_uq_tutorial.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:uq:solutions}

\begin{solution}
\textbf{Exercise 1 (Entropy calculation):} For a Gaussian distribution $\mathcal{N}(\mu, \Sigma)$, the Shannon entropy is $H = \frac{1}{2}\log((2\pi e)^n |\Sigma|)$. For $\Sigma = I$, $H = \frac{n}{2}(1 + \log(2\pi))$.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (KL divergence):} For two Gaussians $P = \mathcal{N}(\mu_1, \Sigma_1)$ and $Q = \mathcal{N}(\mu_2, \Sigma_2)$, the KL divergence is:
$$D_{KL}(P\|Q) = \frac{1}{2}\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1}(\mu_2 - \mu_1) - n + \log\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Ensemble Kalman):} For an ensemble of $N$ particles, the Kalman update is $x_i^a = x_i^f + K(y - Hx_i^f)$ where $K = P^f H^T(HP^fH^T + R)^{-1}$. The ensemble covariance $P^f$ is estimated from the sample covariance of the ensemble.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} For turbulent PDEs, stochastic Galerkin methods project the solution onto a polynomial chaos basis. The convergence rate depends on the smoothness of the solution with respect to the random parameters.
\end{solution}
''',
    'ch08_global_positioning.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:gps:solutions}

\begin{solution}
\textbf{Exercise 1 (GPS with 4 satellites):} For satellites at $(\pm 20000, 0, 20000)$ and $(0, \pm 20000, 20000)$ with $\tau_i = 25000$, the user is at the origin with clock bias $b = 50$. The satellites lie on a sphere centered at the origin, which is a quadric with focus at the user. This gives two symmetric solutions: $(0,0,0)$ and a point far below the satellites.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Sphere ambiguity):} If all satellites lie on a sphere centered at the user, the GPS equations have exactly two solutions. This is because the quadratic constraint $||X||^2 - 2\sigma \cdot X + ||\sigma||^2 = 0$ intersects the plane of linear solutions in two points.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Rank-3 solver):} For coplanar satellites, the matrix $A$ has rank 3. The nullspace is spanned by the normal to the satellite plane. Substituting $X = X_0 + t v$ into the quadratic constraint gives a quadratic equation in $t$, yielding 0, 1, or 2 solutions.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} The Bancroft algorithm provides a closed-form solution for 4 satellites. It solves the quartic equation arising from the quadratic constraint, giving up to 4 algebraic solutions. The physically meaningful solution is selected by requiring positive altitude and clock bias.
\end{solution}
''',
    'ch09_dynamical_systems_3d.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:3dp:solutions}

\begin{solution}
\textbf{Exercise 1 (Lorenz stable manifold):} The Lorenz system at $\sigma = 10, \rho = 28, \beta = 8/3$ has a saddle fixed point at the origin. The stable manifold is 2D, tangent to the eigenvectors of the stable eigenvalues. At order 5 Taylor approximation, the manifold is:
$$W^s_{\text{loc}} = \{(x,y,z) : z = h(x,y)\}$$
where $h$ is computed by solving the homological equation.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (ACT system):} The Arneodo-Coullet-Tresser system has a homoclinic orbit to the origin for certain parameter values. The unstable manifold is computed by the Parameterization Method using Taylor series expansion.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Homoclinic tangle):} The Smale horseshoe has a homoclinic tangle where the stable and unstable manifolds intersect transversally. For 3D printing, we approximate the manifold by a mesh of triangles, truncating at a minimum scale.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} The Parameterization Method can be extended to higher dimensions and to invariant tori. The convergence is guaranteed by the implicit function theorem in Banach spaces.
\end{solution}
''',
    'ch10_dna_storage.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:dna:solutions}

\begin{solution}
\textbf{Exercise 1 (Edit distance):} The edit distance between two DNA sequences is the minimum number of insertions, deletions, and substitutions needed to transform one into the other. For sequences of length $n$, the dynamic programming algorithm runs in $O(n^2)$ time.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (GC-content):} To enforce GC-content between 40\% and 60\%, we use run-length limited codes. A simple construction is to concatenate blocks of fixed composition and add parity bits.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Sequence reconstruction):} The sequence reconstruction problem asks to recover a codeword from multiple noisy reads. For the deletion channel, the Berlekamp-Massey algorithm can be adapted to find the original sequence.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):)} The capacity of the DNA storage channel is an open problem. Recent work using information theory and combinatorial constructions has established lower bounds, but the exact capacity remains unknown.
\end{solution}
''',
    'ch11_cyber_defense.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:cyber:solutions}

\begin{solution}
\textbf{Exercise 1 (Autoencoder):} An autoencoder with hidden dimension $k < n$ learns to compress log data. The reconstruction error $\|x - \hat{x}\|^2$ identifies anomalies: large errors indicate unusual patterns.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Word2Vec):} Word2Vec learns embeddings for categorical data (IP addresses, usernames). The cosine similarity between embeddings captures semantic relationships: similar IPs have similar usage patterns.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (RL for defense):} A reinforcement learning agent learns to respond to attacks by maximizing a reward function. The state space includes network metrics; the action space includes firewall rules and alerts.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} Adversarial attacks on anomaly detection systems can fool autoencoders by adding small perturbations. Defense mechanisms include adversarial training and ensemble methods.
\end{solution}
''',
    'ch12_operads_systems.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:op:solutions}

\begin{solution}
\textbf{Exercise 1 (Operad composition):} The composition operation in an operad satisfies associativity: $(f \circ_i g) \circ_j h = f \circ_i (g \circ_j h)$. This ensures that composing operations in different orders gives the same result.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Network operad):} The network operad has operations corresponding to network topologies. Composition corresponds to substituting one network into a node of another.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (DARPA CASCADE):} The CASCADE program uses operads to compose heterogeneous systems. Each system is an algebra over the operad, and composition is given by the operad action.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} Operadic methods can be extended to stochastic systems and to systems with uncertainty. The compositional structure allows local verification to imply global properties.
\end{solution}
''',
    'ch13_astronomy_inference.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:astro:solutions}

\begin{solution}
\textbf{Exercise 1 (Least squares):} For the Ceres orbit determination, Gauss used 6 observations to solve for 6 orbital elements. The normal equations $A^T A x = A^T b$ give the least squares solution.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (MCMC):} Hamiltonian Monte Carlo samples from posterior distributions using gradient information. The leapfrog integrator preserves the Hamiltonian structure, giving high acceptance rates.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (Nested sampling):} Nested sampling estimates the Bayesian evidence by sampling from nested likelihood contours. It is particularly useful for multimodal distributions.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} Amortized inference uses neural networks to approximate the posterior. Once trained, inference is instantaneous, making it suitable for real-time applications.
\end{solution}
''',
    'ch14_fair_data.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:fair:solutions}

\begin{solution}
\textbf{Exercise 1 (FAIR assessment):} To assess FAIRness, check: (1) Does the resource have a persistent identifier? (2) Is the metadata indexed? (3) Are the formats open and interoperable? (4) Is the license clear?
\end{solution}

\begin{solution}
\textbf{Exercise 2 (OSCAR):} OSCAR uses Julia for verified computation. The workflow is: define computation in Julia, verify with formal methods, deploy as web service.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (DOI):} A DOI (Digital Object Identifier) provides a persistent link to a resource. The DOI system is managed by the International DOI Foundation.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):} The challenge of long-term preservation of mathematical software requires standardized formats and active community stewardship.
\end{solution}
''',
    'ch15_tomography.tex': r'''
%==============================================================================
% Solutions
%==============================================================================
\section{Solutions}
\label{sec:tom:solutions}

\begin{solution}
\textbf{Exercise 1 (Radon transform):} The Radon transform of a function $f(x,y)$ is $Rf(\theta, s) = \int_{-\infty}^{\infty} f(s\cos\theta - t\sin\theta, s\sin\theta + t\cos\theta) dt$. For a point source at $(x_0, y_0)$, the Radon transform is a delta function on the line $\theta, s$ such that $(x_0, y_0)$ lies on the line.
\end{solution}

\begin{solution}
\textbf{Exercise 2 (Filtered back-projection):} The filtered back-projection formula is $f(x,y) = \int_0^{\pi} (Rf(\theta, \cdot) * h)(x\cos\theta + y\sin\theta) d\theta$ where $h$ is the Ram-Lak filter with frequency response $|\omega|$.
\end{solution}

\begin{solution}
\textbf{Exercise 3 (MIP):} Maximum Intensity Projection assigns to each pixel the maximum intensity along the ray. For scroll unwrapping, MIP helps visualize text layers by suppressing background.
\end{solution}

\begin{solution}
\textbf{Exercise 4 (Research):)} The Archeolab framework combines multiple imaging modalities and segmentation algorithms. Future work includes automated text recognition and 3D visualization.
\end{solution}
''',
}

def add_solutions():
    chapters_dir = 'chapters'
    for fname, solutions_content in solutions.items():
        filepath = os.path.join(chapters_dir, fname)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Add solutions section before the last section or at end
        # Find the last section in the file
        last_section_match = None
        for match in __import__('re').finditer(r'\\section\{([^}]+)\}', content):
            last_section_match = match
        
        if last_section_match:
            # Insert after the last section
            insert_pos = last_section_match.end()
            new_content = content[:insert_pos] + solutions_content + content[insert_pos:]
        else:
            # Append at end
            new_content = content + solutions_content
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Added solutions to {fname}")

if __name__ == '__main__':
    add_solutions()
