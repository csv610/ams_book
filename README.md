# Applied Mathematics from the Notices of the AMS

A computational mathematics textbook collecting fifteen articles from the *Notices of the American Mathematical Society* (2000–2025), selected for their value to scientists and engineers. Each chapter combines rigorous theory with runnable code and exercises.

## Overview

| Metric | Value |
|--------|-------|
| Chapters | 15 |
| Pages | 102 |
| Code listings | 30+ |
| Exercises | 50+ |
| Bibliography | 21 entries |
| Index entries | 48 |

## Structure

The book is organized into six parts:

### Part I: Computational Foundations
- **Chapter 1:** Numerical Stability in Gaussian Elimination — John Urschel
- **Chapter 2:** Algorithmic Fewnomial Theory Over R — J. Maurice Rojas
- **Chapter 3:** Spectral Methods in Microeconomics — Benjamin Golub

### Part II: Geometric Machine Learning
- **Chapter 4:** On the Geometry of Deep Learning — Balestriero, Humayun, Baraniuk
- **Chapter 5:** What Is... an Equivariant Neural Network? — Lim, Nelson
- **Chapter 6:** Topology Meets Machine Learning: The Euler Characteristic Transform — Bastian Rieck

### Part III: Uncertainty, Dynamics, and Inverse Problems
- **Chapter 7:** Taming Uncertainty: The Rise of Uncertainty Quantification — Chen, Wiggins, Andreou
- **Chapter 8:** New Developments in Global Positioning — Boutin, Kemper
- **Chapter 9:** 3D Printing of Invariant Manifolds in Dynamical Systems — Bishop et al.

### Part IV: Modern Applications
- **Chapter 10:** Exciting Coding Problems for DNA-based Storage Systems — Bar-Lev, Sabary, Yaakobi
- **Chapter 11:** The Mathematics of Cyber Defense — Emanuello, Ridley
- **Chapter 12:** Operads for Designing Systems of Systems — Baez, Foley

### Part V: Data-Intensive Science
- **Chapter 13:** A Brief History of Inference in Astronomy — de Souza, Ishida, Krone-Martins
- **Chapter 14:** Making Mathematical Online Resources FAIR — Bacher et al.
- **Chapter 15:** Deciphering Scrolls with Tomography: A Training Experiment — Foschiatti, Kittenberger, Scherzer

### Part VI: The Future
- Chapter 16 (planned): Machine-Assisted Proof — Terence Tao

## Building

### Prerequisites
- TeX Live 2023+ (or MacTeX)
- Python 3.10+ with Julia bindings (optional, for companion code)

### Compilation

```bash
# Generate PDF
pdflatex ams_bool.tex
bibtex ams_bool
pdflatex ams_bool.tex
pdflatex ams_bool.tex

# Generate index
makeindex ams_bool.idx
pdflatex ams_bool.tex
pdflatex ams_bool.tex
```

### Companion Code

Julia code is in `code/chXX/` directories. C++ reference implementations are in `code_cpp/chXX/`.

```julia
# Example: Chapter 1
include("code/ch01/gaussian_elimination.jl")

# Example: Chapter 2
include("code/ch02/fewnomial_theory.jl")
```

## Content Summary

Each chapter follows a consistent structure:

1. **Historical context** — original publication details, editorial introduction
2. **Mathematical theory** — definitions, theorems, proofs
3. **Algorithmic perspective** — computational implications, complexity
4. **Code listings** — runnable implementations in Julia/Python/C++
5. **Open problems** — research directions
6. **Exercises** — computational and theoretical tasks

## Target Audience

- Graduate students in applied mathematics, computational science, and engineering
- Researchers seeking a bridge between pure theory and computational practice
- Instructors looking for course materials in numerical linear algebra, geometric ML, uncertainty quantification, or modern applications

## Citation

```bibtex
@book{amsnotices2025,
  title = {Applied Mathematics from the Notices of the AMS},
  editor = {Various AMS Notices Authors},
  year = {2025},
  publisher = {American Mathematical Society},
  note = {Selected articles adapted and edited}
}
```

## License

Articles reprinted with permission from *Notices of the American Mathematical Society*. Original articles remain under their respective copyrights. This compilation is for educational use.

## Repository

- **Source:** https://github.com/csv610/ams_book
- **PDF:** See `ams_bool.pdf`

## Related Projects

- [Math21Century](https://github.com/csv610/math21century) — 30-chapter textbook on 21st-century mathematics
- [Gauss](https://github.com/csv610/gauss_book) — Computational mathematics textbook with 352 tests
