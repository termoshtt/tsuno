//! Initial factorization of a COO matrix into [LU].

use std::collections::BTreeMap;

use crate::lower::UnitTriangle;
use crate::*;
use ndarray::Array2;

const DROP_TOLERANCE: f64 = 1.0e-12;
const PIVOT_THRESHOLD: f64 = 1.0e-12;

/// Workspace for initial factorization of a COO matrix into [LU].
pub struct Worker {
    /// Current step of the factorization process, $k$ in the paper.
    step: usize,

    nrows: usize,
    ncols: usize,

    /// Mutable row-major view of the current work matrix.
    ///
    /// This stores the unfactorized part of the matrix and is updated during
    /// elimination. Fill-ins are inserted here and dropped entries are removed.
    work_rows: Vec<BTreeMap<usize, f64>>,
    /// Mutable column-major view of the current work matrix.
    ///
    /// This mirrors `work_rows` so pivot search and column elimination can find
    /// active non-zeros without scanning every row.
    work_cols: Vec<BTreeMap<usize, f64>>,
    /// Rows still present in the current Schur complement.
    ///
    /// Once a row is selected as a pivot row and stored in `U`, it is marked
    /// inactive and excluded from later pivot search and row updates.
    active_rows: Vec<bool>,
    /// Columns still present in the current Schur complement.
    ///
    /// Once a column is selected as a pivot column and eliminated from active
    /// rows, it is marked inactive and excluded from later pivot search and
    /// Markowitz counts.
    active_cols: Vec<bool>,
    l_units: Vec<UnitTriangle>,
    /// Finalized rows of `U` in factorization order.
    ///
    /// A pivot row is copied here before the corresponding work row is cleared
    /// and marked inactive.
    factorized_u_rows: Vec<Vec<(usize, f64)>>,
    p: Vec<usize>,
    q: Vec<usize>,
}

impl Worker {
    pub fn from_dense(array: Array2<f64>) -> Self {
        let (nrows, ncols) = array.dim();
        let coo = array
            .indexed_iter()
            .filter(|&(_, &value)| value != 0.0)
            .map(|((row, col), &value)| (row, col, value))
            .collect::<Vec<_>>();
        Self::from_coo_matrix(nrows, ncols, coo.into_iter())
    }

    pub fn from_coo_matrix(
        nrows: usize,
        ncols: usize,
        coo: impl Iterator<Item = (usize, usize, f64)>,
    ) -> Self {
        assert!(nrows > 0, "number of rows must be positive");
        assert!(ncols > 0, "number of columns must be positive");

        let entries = coo.collect::<Vec<_>>();
        for &(row, col, _) in &entries {
            assert!(
                row < nrows,
                "row index {row} is out of bounds for {nrows} rows"
            );
            assert!(
                col < ncols,
                "column index {col} is out of bounds for {ncols} columns"
            );
        }

        let mut work_rows = vec![BTreeMap::new(); nrows];
        let mut work_cols = vec![BTreeMap::new(); ncols];
        for (row, col, value) in entries {
            let value = work_rows[row].get(&col).copied().unwrap_or(0.0) + value;
            if value.abs() <= DROP_TOLERANCE {
                work_rows[row].remove(&col);
                work_cols[col].remove(&row);
            } else {
                work_rows[row].insert(col, value);
                work_cols[col].insert(row, value);
            }
        }

        Self {
            step: 0,
            nrows,
            ncols,
            work_rows,
            work_cols,
            active_rows: vec![true; nrows],
            active_cols: vec![true; ncols],
            l_units: Vec::new(),
            factorized_u_rows: Vec::new(),
            p: Vec::new(),
            q: Vec::new(),
        }
    }

    fn active_row_len(&self, row: usize) -> usize {
        self.work_rows[row]
            .keys()
            .filter(|&&col| self.active_cols[col])
            .count()
    }

    fn active_col_len(&self, col: usize) -> usize {
        self.work_cols[col]
            .keys()
            .filter(|&&row| self.active_rows[row])
            .count()
    }

    fn set_entry(&mut self, row: usize, col: usize, value: f64) {
        if value.abs() <= DROP_TOLERANCE {
            self.work_rows[row].remove(&col);
            self.work_cols[col].remove(&row);
        } else {
            self.work_rows[row].insert(col, value);
            self.work_cols[col].insert(row, value);
        }
    }

    /// Choose the next pivot by Markowitz pivoting.
    ///
    /// For an active non-zero `a_ij`, let `r_i` be the number of active
    /// non-zeros in row `i` and `c_j` be the number of active non-zeros in
    /// column `j`. Eliminating column `j` with row `i` can combine every
    /// non-pivot entry in row `i` with every non-pivot entry in column `j`, so
    /// the number of possible fill-ins is bounded by
    ///
    /// ```text
    /// (r_i - 1) * (c_j - 1).
    /// ```
    ///
    /// Markowitz pivoting minimizes this bound among numerically acceptable
    /// pivots. The numerical test here accepts `a_ij` only when
    ///
    /// ```text
    /// |a_ij| >= PIVOT_THRESHOLD * max_k |a_kj|
    /// ```
    ///
    /// over active rows in the same column. Ties are broken by larger pivot
    /// magnitude and then by row/column index for deterministic output.
    fn choose_pivot(&self) -> Option<(usize, usize)> {
        let mut best = None;
        for row in 0..self.nrows {
            if !self.active_rows[row] {
                continue;
            }
            let row_count = self.active_row_len(row);
            if row_count == 0 {
                continue;
            }
            for (&col, &value) in &self.work_rows[row] {
                if !self.active_cols[col] || value.abs() <= DROP_TOLERANCE {
                    continue;
                }
                let col_max = self.work_cols[col]
                    .iter()
                    .filter(|(candidate_row, _)| self.active_rows[**candidate_row])
                    .map(|(_, candidate)| candidate.abs())
                    .fold(0.0, f64::max);
                if value.abs() < PIVOT_THRESHOLD * col_max {
                    continue;
                }
                let col_count = self.active_col_len(col);
                let cost = (row_count - 1) * (col_count - 1);
                let score = (cost, -value.abs(), row, col);
                if best
                    .as_ref()
                    .is_none_or(|(best_score, _, _)| score < *best_score)
                {
                    best = Some((score, row, col));
                }
            }
        }
        best.map(|(_, row, col)| (row, col))
    }

    /// Eliminate the active column `j` with pivot row `i`.
    ///
    /// For the selected pivot `a = A[i, j]`, this step stores the active part
    /// of row `i` as the next row of `U`. Then, for each other active row `k`
    /// with a non-zero entry `b = A[k, j]`, it applies
    ///
    /// ```text
    /// μ = b / a
    /// row_k <- row_k - μ row_i
    /// ```
    ///
    /// so that column `j` is eliminated from row `k`.
    ///
    /// ```text
    ///        j
    /// i | ... a ... v ...
    /// k | ... b ... w ...
    ///
    ///        j
    /// i | ... a ... v ...
    /// k | ... 0 ... w - μ v ...
    /// ```
    ///
    /// Each non-zero `μ` is recorded as a unit triangle factor in `L`.
    ///
    fn unit_factorize(&mut self, i: usize, j: usize) {
        debug_assert!(self.active_rows[i]);
        debug_assert!(self.active_cols[j]);

        let pivot = *self.work_rows[i]
            .get(&j)
            .expect("pivot must exist in the active matrix");
        assert!(pivot.abs() > DROP_TOLERANCE, "pivot must be non-zero");

        let pivot_row = self.work_rows[i]
            .iter()
            .filter(|(col, value)| self.active_cols[**col] && value.abs() > DROP_TOLERANCE)
            .map(|(&col, &value)| (col, value))
            .collect::<Vec<_>>();

        let mut u_row = Vec::with_capacity(pivot_row.len());
        u_row.push((j, pivot));
        u_row.extend(pivot_row.iter().copied().filter(|&(col, _)| col != j));

        let target_rows = self.work_cols[j]
            .iter()
            .filter(|(row, _)| self.active_rows[**row] && **row != i)
            .map(|(&row, &value)| (row, value))
            .collect::<Vec<_>>();

        for (row, value) in target_rows {
            let mu = value / pivot;
            if mu.abs() > DROP_TOLERANCE {
                self.l_units.push(UnitTriangle::new(mu, row, i));
            }

            for &(col, pivot_value) in &pivot_row {
                if col == j {
                    continue;
                }
                let value =
                    self.work_rows[row].get(&col).copied().unwrap_or(0.0) - mu * pivot_value;
                self.set_entry(row, col, value);
            }
            self.set_entry(row, j, 0.0);
        }

        for &(col, _) in &pivot_row {
            self.work_cols[col].remove(&i);
        }
        self.work_rows[i].clear();
        self.active_rows[i] = false;
        self.active_cols[j] = false;
        self.factorized_u_rows.push(u_row);
        self.p.push(i);
        self.q.push(j);
        self.step += 1;
    }

    pub fn factorize(mut self) -> LU {
        while self.step < self.nrows.min(self.ncols) {
            let Some((row, col)) = self.choose_pivot() else {
                break;
            };
            self.unit_factorize(row, col);
        }
        LU {
            nrows: self.nrows,
            ncols: self.ncols,
            l: L::from_units(self.l_units),
            u: U::from_rows(self.factorized_u_rows),
            p: self.p,
            q: self.q,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn from_coo_matrix_builds_workspace_with_empty_rows_and_columns() {
        let matrix = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 20.0, 0.0, 0.0, 0.0],
        ];
        let worker = Worker::from_dense(matrix);

        assert!(worker.work_rows[0].is_empty());
        assert_eq!(
            worker.work_rows[1].iter().collect::<Vec<_>>(),
            vec![(&3, &10.0)]
        );
        assert!(worker.work_rows[2].is_empty());
        assert_eq!(
            worker.work_rows[3].iter().collect::<Vec<_>>(),
            vec![(&1, &20.0)]
        );
        assert_eq!(
            worker.work_cols[1].iter().collect::<Vec<_>>(),
            vec![(&3, &20.0)]
        );
        assert_eq!(
            worker.work_cols[3].iter().collect::<Vec<_>>(),
            vec![(&1, &10.0)]
        );
    }

    #[test]
    #[should_panic(expected = "number of rows must be positive")]
    fn from_coo_matrix_rejects_zero_rows() {
        Worker::from_coo_matrix(0, 1, Vec::new().into_iter());
    }

    #[test]
    #[should_panic(expected = "number of columns must be positive")]
    fn from_coo_matrix_rejects_zero_columns() {
        Worker::from_coo_matrix(1, 0, Vec::new().into_iter());
    }

    #[test]
    fn choose_pivot_prefers_minimum_markowitz_cost() {
        let matrix = array![[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
        let worker = Worker::from_dense(matrix);

        assert_eq!(worker.choose_pivot(), Some((1, 0)));
    }

    #[test]
    fn factorize_reconstructs_rectangular_matrix() {
        let matrix = array![
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 5.0, 0.0],
        ];
        let lu = Worker::from_dense(matrix.clone()).factorize();

        let reconstructed = lu.reconstruct();

        assert_abs_diff_eq!(reconstructed, matrix, epsilon = 1.0e-9);
    }

    #[test]
    fn factorize_reconstructs_tall_matrix() {
        let matrix = array![
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 6.0],
        ];
        let lu = Worker::from_dense(matrix.clone()).factorize();

        let reconstructed = lu.reconstruct();

        assert_abs_diff_eq!(reconstructed, matrix, epsilon = 1.0e-9);
    }

    #[test]
    fn from_dense_factorizes_dense_matrix() {
        let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];

        let lu = LU::from_dense(matrix.clone());
        let reconstructed = lu.reconstruct();

        assert_abs_diff_eq!(reconstructed, matrix, epsilon = 1.0e-9);
    }
}
