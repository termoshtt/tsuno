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
/// Invariant
/// ---------
/// - $\mu_k \neq 0$
/// - $r_k \neq c_k$
/// - Length of `mu`, `col`, and `row` are the same.
///
pub struct L {
    /// Multipliers $\mu_k$
    pub mu: Vec<f64>,
    /// Column $c_k$
    pub col: Vec<usize>,
    /// Row $r_k$
    pub row: Vec<usize>,
}
