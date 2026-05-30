# tsuno
Simplex-based LP solver/analyzer for advanced usages.

## Modules

- `tsuno::lu`: sparse LU factorization and eta-update linear algebra kernels
  for revised simplex methods.
- `tsuno::simplex`: standard-form LP and basis-level building blocks for the
  revised simplex method.

## Current Status

The current implementation is a standard-form LP solver/analyzer for

```text
min c^T x
subject to A x = b
           x >= 0
```

It currently provides:

- Sparse LU factorization for basis matrices, including basis solves for
  `B x = rhs` and `B^T x = rhs`.
- Product-form eta updates for one-column basis replacement.
- Standard-form LP representation as `A`, `b`, and `c`.
- Basis-level revised simplex quantities:
  - basic solution `x_I = B^{-1} b`
  - basis costs `c_I`
  - dual variables `y = B^{-T} c_I`
  - reduced costs `r_j = c_j - A_j^T y`
- Primal revised simplex with Phase I feasible-basis construction and
  `simplex::primal::solve`.
- Dual revised simplex for dual-feasible initial bases.
- Structured simplex traces for solver paths and snapshot tests.
- Farkas certificates for infeasible standard-form LPs.
- Certificate simplification by deletion filter; the support of a simplified
  certificate gives a standard-form row IIS candidate.

The LU update path currently uses eta updates. Forrest-Tomlin-style updates,
stronger numerical pivoting, explicit refactorization, residual checks, and
sparse right-hand-side solve paths are future improvements to the linear algebra
kernel.

## Roadmap

The next goal is to grow from a standard-form solver into an LP analysis toolkit
for LPs supplied as OMMX instances. The solver can work on `StandardFormLp`, but
diagnostics such as IIS need to be mapped back to OMMX constraint and variable
identifiers.

### Sparse LU Kernel

The sparse LU kernel is the performance-critical basis representation used by
the revised simplex variants. Its current product-form eta update path is
simple and testable, but it is not the final performance target.

- [ ] Improve numerical stability in sparse pivot selection.
  - Add threshold pivoting around the current Markowitz-style sparsity
    heuristic.
  - Track growth, small pivots, and residual quality so unstable bases can be
    refactorized.
- [ ] Add explicit basis refactorization from the latest basis matrix.
- [ ] Replace or supplement eta updates with Forrest-Tomlin-style updates to
  keep repeated one-column basis changes cheaper.
- [ ] Add sparse right-hand-side solve paths for both `B x = rhs` and
  `B^T x = rhs`.
- [ ] Add hyper-sparse solve paths for simplex pricing and pivot operations
  where the right-hand side or result remains very sparse.
- [ ] Reduce allocation and data movement in repeated basis solves.
- [ ] Benchmark factorization, solve, transposed solve, and update workloads
  across several sparsity levels.

### Simplex Methods

- [ ] Add optional fast paths for obvious feasible initial bases, such as slack
  bases, so top-level solve does not always need Phase I.
- [ ] Add a top-level dual-simplex entry point that can construct or recover a
  dual-feasible starting basis when useful.
- [ ] Add warm-start and reoptimization APIs for modified LPs.
  - Reuse the current basis when only the right-hand side `b` changes, using
    dual simplex when primal feasibility is broken.
  - Reuse the current basis when only the objective `c` changes, using primal
    simplex when dual feasibility is broken.
  - Reuse basis factorizations when nonbasis columns of `A` change.
  - Refactorize or update the basis representation when basis columns of `A`
    change.
  - Support constraint and variable additions and deletions, including basis
    repair and index-map updates.
- [ ] Expand pivot selection strategies beyond the current deterministic rules.
- [ ] Add more numerical termination checks around residuals and certificate
  quality.

### Certificates And IIS

For the standard-form infeasible system

```text
A x = b
x >= 0
```

a Farkas certificate can be represented by a multiplier `y` satisfying

```text
A^T y >= 0
b^T y < 0
```

which proves that no feasible `x >= 0` can satisfy `A x = b`.

- [ ] Add richer certificate diagnostics for numerical quality without making
  invalid certificates constructible.
- [ ] Add stronger tests for certificate simplification across larger and
  degenerate infeasible systems.
- [ ] Return IIS results in terms of OMMX constraint identifiers through the
  standardization map.

### OMMX Standardization

The project entry point should be an OMMX instance. Since the solver operates on
standard-form LPs, conversion should standardize the instance immediately:

```text
OMMX Instance
  -> StandardFormLp
  + StandardizationMap
```

The map records how standard-form rows and columns came from the OMMX model, so
certificates and IIS supports can be lifted back to caller-facing OMMX IDs.

- [ ] Add an OMMX dependency and parse linear objectives, constraints, and
  variable bounds from OMMX instances.
- [ ] Convert OMMX LP data directly into `StandardFormLp`.
- [ ] Define `StandardizationMap` for OMMX variable IDs, constraint IDs,
  generated rows, and generated columns.
- [ ] Preserve row provenance for inequalities, equalities, and bound-derived
  constraints.
- [ ] Lift Farkas certificates and simplified supports back to caller-facing
  OMMX constraint identifiers.

# License
Copyright (c) 2026 Toshiki Teramura (@termoshtt)

This software is licensed under the [MIT License](./LICENSE-MIT) OR [Apache License Version 2.0](./LICENSE-APACHE) (at your option).
