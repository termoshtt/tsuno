# tsuno-lu
Sparse LU factorizer with update capabilities inspired by LUSOL.

## Planned features

`tsuno-lu` is intended to provide the linear algebra kernel needed by a revised
simplex method without explicitly forming the inverse of the basis matrix.
The public API should use descriptive operation names instead of traditional
Fortran-style abbreviations.

- [x] Build an initial sparse LU representation from COO input.
- [x] Build an initial sparse LU representation from a dense `ndarray`.
- [x] Choose initial pivots with a Markowitz-style fill-in estimate.
- [x] Store the nominal lower factor as a product of unit triangular
  eliminations.
- [x] Store the nominal upper factor as sparse pivot rows.
- [x] Reconstruct the original matrix from the current initial factorization for
  validation.
- [x] Solve a linear system with the represented matrix.
  - Name: `solve`, for `A x = rhs`. In a simplex solver, this is used with the
    current basis matrix `B`.
- [ ] Solve a transposed linear system with the represented matrix.
  - Proposed name: `solve_transposed`, for `A^T x = rhs`. In a simplex solver,
    this is used with the current basis matrix `B`.
- [ ] Add a product-form basis column replacement.
  - Proposed name: `replace_basis_column`, storing the eta column
    `B^{-1} a_q` and the leaving basis position.
- [ ] Apply accumulated column replacements when solving basis systems.
- [ ] Apply accumulated column replacements in reverse order when solving
  transposed basis systems.
- [ ] Report how many delayed basis updates are currently stored.
  - Proposed name: `basis_update_count`.
- [ ] Rebuild the sparse LU representation from the latest explicit basis when
  delayed updates become too expensive or inaccurate.
  - Proposed name: `refactor_basis`.
- [ ] Provide residual checks for basis solves and transposed basis solves.
- [ ] Provide sparse right-hand-side solve paths for basis systems.
- [ ] Provide sparse right-hand-side solve paths for transposed basis systems.
- [ ] Provide row/column access helpers for simplex pricing operations.
- [ ] Add Forrest-Tomlin-style updates to reduce product-form update growth.
