//! LU decomposition with dynamic update for non-square sparse matrices.

mod eta_update;
mod initial_factorize;
mod lower;
mod solve;
mod solve_transposed;
#[cfg(test)]
mod test_support;
mod upper;

use ndarray::Array2;

pub use eta_update::*;
pub use initial_factorize::*;
pub use lower::*;
pub use upper::*;

#[katexit::katexit]
/// Storage for a LU-decomposed matrix
///
/// This library factorize a sparse matrix $A$ into $A = LU$ (not $PA = LU$ or $PAQ = LU$) with _nominal_ LU decomposition.
/// Strictly speaking, $L$ is not necessarily lower-triangular, and $U$ is not necessarily upper-triangular.
/// $L$ is the product of unit triangle matrices which is not necessarily lower-triangular,
/// and row and column permutations $P$ and $Q$ are managed to keep $PUQ$ upper-triangular or trapezoidal.
///
/// Column replacement updates
/// --------------------------
///
/// After the initial factorization, this type can represent later one-column
/// replacements by accumulating product-form eta updates. If the represented
/// matrix is currently $A_k$ and column $p$ is replaced by $a_{\mathrm{new}}$,
/// first compute
///
/// $$
/// d = A_k^{-1} a_{\mathrm{new}}.
/// $$
///
/// Then the updated matrix can be written as
///
/// $$
/// A_{k+1} = A_k E_k,
/// $$
///
/// where $E_k$ is the identity matrix with its $p$-th column replaced by
/// $d$:
///
/// $$
/// E_k =
/// \begin{bmatrix}
/// e_0 & \cdots & d & \cdots & e_{n-1}
/// \end{bmatrix}.
/// $$
///
/// Therefore, after several updates,
///
/// $$
/// A_k = A_0 E_0 E_1 \cdots E_{k-1}.
/// $$
///
/// A call to [`LU::solve`] applies
///
/// $$
/// A_k^{-1}
/// = E_{k-1}^{-1} \cdots E_1^{-1} E_0^{-1} A_0^{-1}
/// $$
///
/// without forming any inverse explicitly. For an eta column $d$ with pivot
/// position $p$, applying $E^{-1}$ to a vector $v$ only needs that column:
///
/// $$
/// t = \frac{v_p}{d_p},
/// \qquad
/// v_i \leftarrow v_i - d_i t \quad (i \ne p),
/// \qquad
/// v_p \leftarrow t.
/// $$
///
/// A call to [`LU::solve_transposed`] applies the transpose-side updates in the
/// opposite order:
///
/// $$
/// A_k^{-T}
/// = A_0^{-T} E_0^{-T} E_1^{-T} \cdots E_{k-1}^{-T}.
/// $$
///
/// Applying $E^{-T}$ leaves non-pivot entries unchanged and updates the pivot
/// entry as
///
/// $$
/// v_p \leftarrow
/// \frac{
///     v_p - \sum_{i \ne p} d_i v_i
/// }{d_p}.
/// $$
///
/// This eta representation is the first update mechanism used by this crate.
/// The long-term plan is to replace or supplement it with Forrest--Tomlin-style
/// updates once the product-form update path is fully established.
pub struct LU {
    nrows: usize,
    ncols: usize,
    l: L,
    u: U,
    /// Row permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    p: Vec<usize>,
    /// Column permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    q: Vec<usize>,
    eta_updates: Vec<EtaUpdate>,
}

impl LU {
    /// Initial factorization from a COO matrix
    pub fn initial_factorize(
        nrows: usize,
        ncols: usize,
        coo: impl Iterator<Item = (usize, usize, f64)>,
    ) -> Self {
        Worker::from_coo_matrix(nrows, ncols, coo).factorize()
    }

    /// Initial factorization from a dense matrix.
    pub fn from_dense(array: Array2<f64>) -> Self {
        Worker::from_dense(array).factorize()
    }

    pub fn l(&self) -> &L {
        &self.l
    }

    pub fn u(&self) -> &U {
        &self.u
    }

    pub fn row_permutation(&self) -> &[usize] {
        &self.p
    }

    pub fn col_permutation(&self) -> &[usize] {
        &self.q
    }

    pub fn eta_updates(&self) -> &[EtaUpdate] {
        &self.eta_updates
    }

    pub fn update_count(&self) -> usize {
        self.eta_updates.len()
    }

    pub fn reconstruct(&self) -> Array2<f64> {
        let mut matrix = Array2::zeros((self.nrows, self.ncols));
        for (step, row) in self.u.rows().enumerate() {
            for (col, value) in row {
                matrix[(self.p[step], col)] = value;
            }
        }
        for (mu, row, col) in self.l.units().collect::<Vec<_>>().into_iter().rev() {
            let pivot_row = matrix.row(col).to_owned();
            for entry_col in 0..self.ncols {
                matrix[(row, entry_col)] += mu * pivot_row[entry_col];
            }
        }
        for eta_update in &self.eta_updates {
            let column = eta_update.pivot();
            let replacement = matrix.dot(eta_update.column());
            matrix.column_mut(column).assign(&replacement);
        }
        matrix
    }
}

pub(crate) fn assert_solve_ready(lu: &LU) {
    assert_eq!(lu.nrows, lu.ncols, "solve requires a square matrix");
    assert_eq!(
        lu.p.len(),
        lu.nrows,
        "solve requires a full-rank factorization"
    );
}
