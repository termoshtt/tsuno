# tsuno

## Educational Project - Not For Production

`tsuno` is an educational Rust project for learning and experimenting with
linear-programming algorithms, especially revised simplex methods and basis
factorization updates.

Do not use this crate in production. It is not a numerically hardened optimizer,
its API is still evolving, and it does not provide the robustness, validation,
presolve, scaling, benchmarking, or operational guarantees expected from a
production LP solver.

## What It Is

`tsuno` is a simplex-based LP solver and analysis library. The current solver
works on standard-form linear programs:

$$
\begin{aligned}
\min_x \quad & c^\top x \\
\text{subject to} \quad & A x = b, \\
& x \ge 0.
\end{aligned}
$$

The main purpose is to make simplex internals explicit and inspectable:

- basis construction and basis-level algebra,
- sparse LU factorization and eta-update mechanics,
- primal and dual revised simplex steps,
- warm-start and reoptimization behavior after small LP edits,
- Farkas certificates and row-support simplification for infeasible LPs,
- structured traces of simplex iterations.

## Quick Example

Add `tsuno` and `ndarray` to a Rust project:

```toml
[dependencies]
tsuno = { git = "https://github.com/termoshtt/tsuno" }
ndarray = "0.16"
```

Then solve a standard-form LP:

```rust
use ndarray::array;
use tsuno::simplex::{
    NoTrace, RevisedSimplexOptions, SimplexResult, StandardFormLp, primal,
};

fn main() {
    // min -x0 - 2*x1
    // s.t. x0      + x2      = 4
    //           x1      + x3 = 3
    //      x >= 0
    let lp = StandardFormLp::new(
        array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        array![4.0, 3.0],
        array![-1.0, -2.0, 0.0, 0.0],
    )
    .expect("valid standard-form LP");

    let mut trace = NoTrace;
    let result = primal::solve(lp, RevisedSimplexOptions::default(), &mut trace)
        .expect("simplex solve");

    match result {
        SimplexResult::Optimal(solution) => {
            println!("objective = {}", solution.objective_value);
            println!("x = {:?}", solution.primal);
            println!("basis = {:?}", solution.basis_indices);
        }
        other => {
            println!("non-optimal result: {other:?}");
        }
    }
}
```

## Documentation

API documentation is published from `cargo doc` on GitHub Pages:
<https://termoshtt.github.io/tsuno/>

## Current Features

### `tsuno::simplex`

- `StandardFormLp` for $A$, $b$, and $c$ in standard-form minimization.
- `primal::solve`, a top-level primal revised simplex entry point with Phase I
  feasible-basis construction.
- Low-level `primal::RevisedSimplex` for stepping from an explicit
  primal-feasible basis.
- `dual::DualRevisedSimplex` for stepping from an explicit dual-feasible basis.
- `warm_start` and `SolvedSimplex` for reusing a basis after supported LP edits:
  replacing $b$, replacing $c$, replacing columns, adding columns, and removing
  columns, plus adding less-than-or-equal constraints with slack variables.
- `FarkasCertificate` for standard-form infeasibility proofs.
- `FarkasCertificate::deletion_filter` for simplifying a certificate support
  into a smaller standard-form row subsystem.
- `NoTrace` and `FullTrace` for either ignoring or recording solver paths.

### `tsuno::lu`

- Sparse LU factorization from COO data or dense `ndarray` matrices.
- Basis solves for $B x = \mathrm{rhs}$ and $B^\top x = \mathrm{rhs}$.
- Product-form eta updates for one-column basis replacement.
- Block basis solves for adding less-than-or-equal constraints with a slack
  basis variable.

## Current Limitations

This project intentionally remains small and experimental.

- It only solves LPs already expressed in standard form.
- It is not a modeling layer and does not currently parse OMMX, MPS, LP, or
  other external problem formats.
- It does not solve MILP, QP, conic, nonlinear, or mixed-domain problems.
- Numerical handling is incomplete: pivoting, residual checks, refactorization
  strategy, scaling, degeneracy handling, and certificate-quality checks need
  more work.
- Performance is not production-tuned or benchmarked across realistic workloads.
- Public APIs may change as the educational examples and solver internals evolve.

For serious optimization work, use a maintained production solver instead.

## Roadmap

### Sparse LU Kernel

- [ ] Improve numerical stability in sparse pivot selection.
- [ ] Track growth, small pivots, and residual quality.
- [ ] Add explicit basis refactorization from the latest basis matrix.
- [ ] Replace or supplement eta updates with Forrest-Tomlin-style updates.
- [ ] Add sparse and hyper-sparse right-hand-side solve paths.
- [ ] Reduce allocation and data movement in repeated basis solves.
- [ ] Benchmark factorization, solve, transposed solve, and update workloads.

### Simplex Methods

- [ ] Add fast paths for obvious feasible initial bases, such as slack bases.
- [ ] Add a top-level dual-simplex entry point that can construct or recover a
  dual-feasible starting basis.
- [x] Add a refactorization-based warm-start entry point for modified LPs.
- [x] Reoptimize after replacing the right-hand side $b$.
- [x] Reoptimize after replacing the objective $c$.
- [x] Reoptimize after replacing, adding, or removing columns.
- [x] Reoptimize after adding a less-than-or-equal constraint with a slack
  basis variable.
- [ ] Expand pivot selection strategies beyond the current deterministic rules.
- [ ] Add stronger numerical termination checks around residuals and certificate
  quality.

### Certificates And IIS

For the standard-form infeasible system

$$
A x = b,\qquad x \ge 0,
$$

a Farkas certificate can be represented by a multiplier $y$ satisfying

$$
A^\top y \ge 0,\qquad b^\top y < 0,
$$

which proves that no feasible $x \ge 0$ can satisfy $A x = b$.

- [ ] Add richer certificate diagnostics for numerical quality without making
  invalid certificates constructible.
- [ ] Add stronger tests for certificate simplification across larger and
  degenerate infeasible systems.
- [ ] Return IIS results in terms of caller-facing constraint identifiers once a
  standardization layer exists.

### Future Input Standardization

The current solver entry point is `StandardFormLp`. A future analysis workflow
may accept richer LP inputs and standardize them immediately:

```text
Input model
  -> StandardFormLp
  + StandardizationMap
```

The map would record how standard-form rows and columns came from the input
model, so certificates and IIS supports can be lifted back to caller-facing
constraint and variable identifiers.

## License

Copyright (c) 2026 Toshiki Teramura (@termoshtt)

This software is licensed under the [MIT License](./LICENSE-MIT) OR
[Apache License Version 2.0](./LICENSE-APACHE), at your option.
