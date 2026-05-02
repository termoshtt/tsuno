//! LU decomposition with dynamic update for non-square sparse matrices.

mod initial_factorize;
mod lower;
mod upper;

use ndarray::{Array1, Array2};

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

    /// Solve a linear system with the represented matrix.
    ///
    /// This computes `x` in `A x = rhs` using the initial sparse LU
    /// factorization, without explicitly forming `A^{-1}`.
    pub fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_solve_ready(self);
        assert_eq!(
            rhs.len(),
            self.nrows,
            "right-hand side length must match the matrix row dimension"
        );

        let mut transformed_rhs = rhs.to_owned();
        for (mu, row, col) in self.l.units() {
            transformed_rhs[row] -= mu * transformed_rhs[col];
        }

        let mut pivot_rhs = Array1::zeros(self.nrows);
        for (step, &row) in self.p.iter().enumerate() {
            pivot_rhs[step] = transformed_rhs[row];
        }

        let pivot_rows = self
            .u
            .rows()
            .map(|row| row.collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut solution = Array1::zeros(self.ncols);
        for step in (0..pivot_rows.len()).rev() {
            let row = &pivot_rows[step];
            let (pivot_col, pivot) = row[0];
            let known_sum = row[1..]
                .iter()
                .map(|&(col, value)| value * solution[col])
                .sum::<f64>();
            solution[pivot_col] = (pivot_rhs[step] - known_sum) / pivot;
        }

        solution
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

fn assert_solve_ready(lu: &LU) {
    assert_eq!(lu.nrows, lu.ncols, "solve requires a square matrix");
    assert_eq!(
        lu.p.len(),
        lu.nrows,
        "solve requires a full-rank factorization"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn solve_solves_dense_rhs() {
        let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let expected_solution = array![1.0, 2.0, 5.0];
        let rhs = matrix.dot(&expected_solution);
        let lu = LU::from_dense(matrix.clone());

        let solution = lu.solve(&rhs);

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn solve_handles_permuted_pivots() {
        let matrix = array![[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 6.0]];
        let expected_solution = array![3.0, 2.0, 4.0];
        let rhs = matrix.dot(&expected_solution);
        let lu = LU::from_dense(matrix.clone());

        let solution = lu.solve(&rhs);

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    #[should_panic(expected = "solve requires a square matrix")]
    fn solve_rejects_rectangular_matrix() {
        let matrix = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let lu = LU::from_dense(matrix);

        lu.solve(&array![1.0, 2.0]);
    }
}
