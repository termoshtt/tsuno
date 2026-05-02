//! LU decomposition with dynamic update for non-square sparse matrices.

mod initial_factorize;
mod lower;
mod upper;

use ndarray::Array2;

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
pub struct LU {
    nrows: usize,
    ncols: usize,
    l: L,
    u: U,
    /// Row permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    p: Vec<usize>,
    /// Column permutation for $U$, keeping $PUQ$ upper-triangular or trapezoidal.
    q: Vec<usize>,
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
        matrix
    }
}
