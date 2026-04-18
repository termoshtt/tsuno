//! LU decomposition with dynamic update for non-square sparse matrices.

#[katexit::katexit]
/// Storage for L (lower-triangle) matrix of LU decomposition.
///
/// $L$ matrix is represented as a product of unit triangle matrices $L = M_0 M_1 \cdots$,
/// where each unit triangle matrix is
///
/// $$
/// M_k = 1 - \mu_k |r_k\rangle \langle c_k|, \quad r_k \neq c_k
/// $$
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
/// - $\mu_k \neq 0$
/// - $r_k \neq c_k$
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
pub struct U {
    arena: Vec<NonZeroEntry>,
    rows: Vec<RowPtr>,
}

/// Pointer of arena for each row in [U].
///
/// Since [U] allows non-zero entries to be non-contiguous, we need to store the offset and length of each row in the arena.
///
struct RowPtr {
    offset: usize,
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
/// Given matrix $A$ is decomposed into $A = LU$ with [L] and [U] storages.
/// $A$ can be non-square.
///
pub struct LU {
    l: L,
    u: U,
    /// Row permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    p: Vec<usize>,
    /// Column permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    q: Vec<usize>,
}
