# tsuno
Simplex-based LP solver/analyzer for advanced usages.

## Modules

- `tsuno::lu`: sparse LU factorization and eta-update linear algebra kernels
  for revised simplex methods.
- `tsuno::simplex`: standard-form LP and basis-level building blocks for the
  revised simplex method.

## `tsuno::lu` Planned Features

`tsuno::lu` provides the linear algebra kernel needed by a revised simplex
method without explicitly forming the inverse of the basis matrix. The public
API uses descriptive operation names instead of traditional Fortran-style
abbreviations.

- [x] Build an initial sparse LU representation from COO input.
- [x] Build an initial sparse LU representation from a dense `ndarray`.
- [x] Choose initial pivots with a Markowitz-style fill-in estimate.
- [x] Store the nominal lower factor as a product of unit triangular
  eliminations.
- [x] Store the nominal upper factor as sparse pivot rows.
- [x] Reconstruct the original matrix from the current initial factorization for
  validation.
- [x] Solve a linear system with the represented matrix.
  - Name: `solve(&Array1<f64>) -> Array1<f64>`, for `A x = rhs`. In a simplex
    solver, this is used with the current basis matrix `B`.
- [x] Solve a transposed linear system with the represented matrix.
  - Name: `solve_transposed(&Array1<f64>) -> Array1<f64>`, for
    `A^T x = rhs`. In a simplex solver, this is used with the current basis
    matrix `B`.
- [x] Add a product-form basis column replacement.
  - Name: `replace_column`, returning `Result<(), UpdateError>` after storing
    the eta column `A^{-1} a_new` and the replaced column position. In a
    simplex solver, `A` is the current basis matrix `B`.
  - Initial implementation: store product-form eta updates. This keeps the
    update logic simple while the solve and transposed-solve APIs are still
    being built out.
- [x] Apply accumulated column replacements when solving basis systems.
- [x] Apply accumulated column replacements in reverse order when solving
  transposed basis systems.
- [x] Report how many delayed basis updates are currently stored.
  - Name: `update_count`.
- [x] Report whether accumulated delayed updates have reached a refactor
  threshold.
  - Name: `should_refactor`.
- [ ] Rebuild the sparse LU representation from the latest explicit basis when
  delayed updates become too expensive or inaccurate.
  - Proposed name: `refactor_basis`.
- [ ] Provide residual checks for basis solves and transposed basis solves.
- [ ] Provide sparse right-hand-side solve paths for basis systems.
- [ ] Provide sparse right-hand-side solve paths for transposed basis systems.
- [ ] Add Forrest-Tomlin-style updates to reduce product-form update growth.
  - Long-term direction: replace or supplement accumulated eta updates with a
    Forrest-Tomlin representation once the product-form update path is correct
    and covered by tests.

## `tsuno::simplex` Planned Features

`tsuno::simplex` provides the LP-side structures that sit above the LU kernel:
standard-form problem data, basis ownership, pricing quantities, and eventually
the revised simplex iteration loop.

- [x] Represent a standard-form LP as `A`, `b`, and `c`.
  - Name: `StandardFormLp`, for `min c^T x` subject to `A x = b`, `x >= 0`.
- [x] Build a basis representation from a standard-form matrix and basis
  column indices.
  - Name: `Basis`.
- [x] Solve basis systems through the LU-backed basis representation.
  - Name: `Basis::solve`, for `B x = rhs`.
- [x] Solve transposed basis systems through the LU-backed basis representation.
  - Name: `Basis::solve_transposed`, for `B^T x = rhs`.
- [x] Replace one basis column after a pivot.
  - Name: `Basis::replace_column`.
- [x] Return a constraint matrix column for pricing and basis replacement.
  - Name: `StandardFormLp::column`, returning `A_j`.
- [x] Compute the basis cost vector.
  - Name: `StandardFormLp::basis_costs`, returning `c_I`.
- [x] Compute dual variables for a basis.
  - Name: `StandardFormLp::dual_variables`, returning `y = B^{-T} c_I`.
- [x] Compute the reduced cost of a single column.
  - Name: `StandardFormLp::reduced_cost`, returning `r_j = c_j - A_j^T y`.
- [x] Return the nonbasis column indices.
  - Name: `StandardFormLp::nonbasis_indices`.
- [x] Compute reduced costs for all nonbasis columns.
  - Name: `StandardFormLp::reduced_costs`.
- [x] Select an entering column from reduced costs.
  - Name: `StandardFormLp::entering_column`, returning the most negative
    nonbasis reduced cost below a tolerance.
- [x] Compute the current basic solution.
  - Name: `StandardFormLp::basic_solution`, returning `x_I = B^{-1} b`.
- [x] Represent the current revised simplex state.
  - Name: `RevisedSimplex`, owning `StandardFormLp`, the current `Basis`, and
    simplex options.
- [x] Compute a pivot direction for an entering column.
  - Internal operation in `RevisedSimplex::step`, computing `d = B^{-1} A_q`.
- [x] Select a leaving basis position with a ratio test.
  - Internal operation in `RevisedSimplex::step`.
- [x] Apply one primal simplex pivot by updating the basis and bookkeeping.
  - Name: `RevisedSimplex::step`.
- [x] Represent iteration outcomes such as optimal, unbounded, and pivoted.
  - Name: `SimplexStep`.
- [x] Solve by repeatedly applying revised simplex steps.
  - Name: `RevisedSimplex::solve`.
- [x] Represent solve outcomes and optimal solutions.
  - Names: `SimplexSolveResult` and `SimplexSolution`.
- [x] Implement Phase I feasible-basis construction.
  - Name: `find_feasible_basis`, building and solving an auxiliary LP with
    artificial variables.
- [x] Provide a top-level standard-form solve path that does not require callers
  to provide an initial basis.
  - Name: `solve`, running Phase I first and then Phase II primal simplex.
- [x] Represent top-level outcomes including infeasibility discovered by Phase
  I.
  - Name: `SimplexResult`.

# License
Copyright (c) 2026 Toshiki Teramura (@termoshtt)

This software is licensed under the [MIT License](./LICENSE-MIT) OR [Apache License Version 2.0](./LICENSE-APACHE) (at your option).
