# tsuno
Simplex-based LP solver/analyzer for advanced usages.

## Modules

- `tsuno::lu`: sparse LU factorization and eta-update linear algebra kernels
  for revised simplex methods.
- `tsuno::simplex`: standard-form LP and basis-level building blocks for the
  revised simplex method.

## Current Status

The current implementation is centered on standard-form LPs

```text
min c^T x
subject to A x = b
           x >= 0
```

and provides the following pieces.

- [x] Sparse LU factorization for basis matrices.
- [x] Basis solves for `B x = rhs` and `B^T x = rhs`.
- [x] Product-form eta updates for one-column basis replacement.
- [x] Standard-form LP representation as `A`, `b`, and `c`.
- [x] Basis-level simplex quantities:
  - basic solution `x_I = B^{-1} b`
  - basis costs `c_I`
  - dual variables `y = B^{-T} c_I`
  - reduced costs `r_j = c_j - A_j^T y`
- [x] Primal revised simplex step and solve loop.
- [x] Phase I auxiliary problem with artificial variables.
- [x] `simplex::primal::solve` that runs Phase I and then Phase II.
- [x] Structured simplex traces for solver paths and snapshot tests.
- [x] Dual revised simplex step and solve loop for dual-feasible bases.
- [x] Farkas certificates for infeasible standard-form LPs.
- [x] Farkas-certificate-driven deletion-filter IIS construction for
  standard-form row subsystems.

The LU update path currently uses eta updates. Forrest-Tomlin-style updates,
stronger numerical pivoting, explicit refactorization, residual checks, and
sparse right-hand-side solve paths are future improvements to the linear algebra
kernel.

## Roadmap

The project goal is to grow from a primal revised simplex implementation into
an LP analysis toolkit that can also explain infeasibility.

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

### Primal Revised Simplex

- [x] Implement basis-level revised simplex operations.
- [x] Implement Phase I feasible-basis construction.
- [x] Implement top-level primal solve for standard-form LPs.
- [x] Split primal-specific step types under `simplex::primal`.
- [ ] Add optional fast paths for obvious feasible initial bases, such as slack
  bases, so top-level solve does not always need Phase I.

### Dual Revised Simplex

Dual simplex keeps dual feasibility and repairs primal infeasibility. It should
reuse the existing basis, basis solve, transposed solve, reduced-cost, and trace
infrastructure.

- [x] Represent a dual revised simplex solver state.
- [x] Select a leaving basis position from negative basic variables.
- [x] Compute the pivot row via a transposed basis solve.
- [x] Select an entering nonbasis column with the dual minimum ratio test.
- [x] Implement one dual simplex pivot step.
- [x] Place dual-specific step types under `simplex::dual`.
- [x] Implement the dual simplex solve loop and result type.
- [x] Share common basis-state and trace concepts with primal simplex where the
  API remains clear.

### Farkas Certificates

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

- [x] Add a `FarkasCertificate` type for standard-form LPs.
- [x] Make `FarkasCertificate` own the LP it certifies and preserve the
  certificate invariant at construction time.
- [x] Add certificate support extraction for standard-form row indices.
- [x] Return a certificate from Phase I infeasible results.
- [x] Add tests that validate certificates produced by Phase I.
- [x] Use deletion filter to simplify Farkas certificates.

### IIS Construction

An irreducible infeasible subsystem (IIS) is a minimal set of constraints that
is already infeasible. This is an analysis feature rather than just a solver
status, so it needs enough modeling information to explain infeasibility back
to the caller.

- [x] Implement a deletion-filter IIS algorithm for standard-form row
  subsystems.
- [x] Represent IIS extraction as certificate simplification followed by
  support extraction.
- [x] Use Farkas certificates as infeasibility witnesses.
- [x] Expose IIS construction as additional analysis from a Farkas certificate,
  rather than as a direct operation on arbitrary LPs.
- [ ] Preserve mappings from a higher-level LP into standard form.
- [ ] Return IIS results in terms of the caller-facing constraint identifiers.

# License
Copyright (c) 2026 Toshiki Teramura (@termoshtt)

This software is licensed under the [MIT License](./LICENSE-MIT) OR [Apache License Version 2.0](./LICENSE-APACHE) (at your option).
