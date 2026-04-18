//! LU decomposition with dynamic update for non-square sparse matrices.

#[katexit::katexit]
/// Storage for the nominally lower-triangle matrix $L$
///
/// $L$ matrix is represented as a product of unit triangle matrices $L = M_0 M_1 \cdots$,
/// where each unit triangle matrix is
///
/// $$
/// M_k = 1 - \mu_k |r_k\rangle \langle c_k|, \quad r_k \neq c_k
/// $$
///
/// Note that $r_k$ and $c_k$ are just not equal, but they can be in any order.
/// This means that $L$ is not necessarily lower-triangular.
///
pub struct L {
    units: Vec<UnitTriangle>,
}

#[katexit::katexit]
/// Unit triangle matrix in the product representation of [L].
///
/// $$
/// M = 1 - \mu |r\rangle \langle c|, \quad r \neq c
/// $$
///
/// Invariant
/// ---------
/// - $\mu \neq 0$
/// - $r \neq c$
///
struct UnitTriangle {
    mu: f64,
    col: usize,
    row: usize,
}

#[katexit::katexit]
/// Storage for U (upper-triangle or trapezoidal) matrix of LU decomposition.
///
/// $U$ is stored similarly to CSR (Compressed Sparse Row) format, but the non-zero entries are non-contiguous for dynamic update.
///
/// Invariant
/// ----------
/// - The first non-zero entry in each row is the pivot, and the pivot is non-zero.
///
pub struct U {
    arena: Vec<NonZeroEntry>,
    rows: Vec<RowPtr>,
}

/// Pointer of arena for each row in [U].
///
/// Since [U] allows non-zero entries to be non-contiguous, we need to store the offset and length of each row in the arena.
///
struct RowPtr {
    /// Offset of the first non-zero entry in the arena for the row.
    offset: usize,
    /// Number of non-zero entries in the row. `length == 0` means the row is empty, so the matrix is singular.
    length: usize,
}

/// Non-zero entry in with its column index in [U].
struct NonZeroEntry {
    value: f64,
    col: usize,
}

#[katexit::katexit]
/// Storage for LU decomposition of a matrix.
///
/// Given sparse non-square matrix $A$ is decomposed into $A = LU$ with [L] and [U] storages.
/// This library factorize given sparse matrix $A$ into $A = LU$ (not $PA = LU$ or $PAQ = LU$) with _nominal_ LU decomposition.
/// $L$ is the product of unit triangle matrices which is not necessarily lower-triangular,
/// and $PUQ$ is kept upper-triangular or trapezoidal with row permutation $P$ and column permutation $Q$.
///
pub struct LU {
    l: L,
    u: U,
    /// Row permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    p: Vec<usize>,
    /// Column permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    q: Vec<usize>,
}
