use ndarray::Array1;

use super::{LU, LuError, validate_solve_ready};

impl LU {
    /// Solve a transposed linear system with the represented matrix.
    ///
    /// This computes `x` in `A^T x = rhs` using the sparse LU factorization,
    /// without explicitly forming `A^{-T}`.
    pub fn solve_transposed(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, LuError> {
        self.solve_initial_transposed(rhs)
    }

    pub(crate) fn solve_initial_transposed(
        &self,
        rhs: &Array1<f64>,
    ) -> Result<Array1<f64>, LuError> {
        validate_solve_ready(self)?;
        if rhs.len() != self.ncols {
            return Err(LuError::RightHandSideLengthMismatch {
                expected: self.ncols,
                actual: rhs.len(),
            });
        }

        let pivot_rows = self
            .u
            .rows()
            .map(|row| row.collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut pivot_solution = Array1::zeros(self.nrows);
        for step in 0..pivot_rows.len() {
            let row = &pivot_rows[step];
            let (pivot_col, pivot) = row[0];
            let known_sum = pivot_rows[..step]
                .iter()
                .enumerate()
                .map(|(previous_step, previous_row)| {
                    previous_row
                        .iter()
                        .find(|&&(col, _)| col == pivot_col)
                        .map(|&(_, value)| value * pivot_solution[previous_step])
                        .unwrap_or(0.0)
                })
                .sum::<f64>();
            pivot_solution[step] = (rhs[pivot_col] - known_sum) / pivot;
        }

        let mut solution = Array1::zeros(self.nrows);
        for (step, &row) in self.p.iter().enumerate() {
            solution[row] = pivot_solution[step];
        }
        for (mu, row, col) in self.l.units().collect::<Vec<_>>().into_iter().rev() {
            solution[col] -= mu * solution[row];
        }

        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::super::test_support::{diagonally_dominant_matrix, vector};
    use super::super::{LU, LuError};

    #[test]
    fn solve_transposed_solves_dense_rhs() {
        let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let expected_solution = array![1.0, 2.0, 5.0];
        let rhs = matrix.t().dot(&expected_solution);
        let lu = LU::from_dense(matrix).unwrap();

        let solution = lu.solve_transposed(&rhs).unwrap();

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn solve_transposed_handles_permuted_pivots() {
        let matrix = array![[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 6.0]];
        let expected_solution = array![3.0, 2.0, 4.0];
        let rhs = matrix.t().dot(&expected_solution);
        let lu = LU::from_dense(matrix).unwrap();

        let solution = lu.solve_transposed(&rhs).unwrap();

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn solve_transposed_handles_generated_matrices_with_different_sparsity() {
        let mut rng = StdRng::seed_from_u64(300);
        for density in [0.0, 0.1, 0.35, 0.7, 1.0] {
            let matrix = diagonally_dominant_matrix(8, density, &mut rng);
            let expected_solution = vector(8, &mut rng);
            let rhs = matrix.t().dot(&expected_solution);
            let lu = LU::from_dense(matrix).unwrap();

            let solution = lu.solve_transposed(&rhs).unwrap();

            assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        }
    }

    #[test]
    fn solve_transposed_rejects_rectangular_matrix() {
        let matrix = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let lu = LU::from_dense(matrix).unwrap();

        let error = lu.solve_transposed(&array![1.0, 2.0, 3.0]).unwrap_err();

        assert_eq!(error, LuError::NonSquareMatrix { nrows: 2, ncols: 3 });
    }

    #[test]
    fn solve_transposed_rejects_wrong_rhs_length() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let lu = LU::from_dense(matrix).unwrap();

        let error = lu.solve_transposed(&array![1.0]).unwrap_err();

        assert_eq!(
            error,
            LuError::RightHandSideLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }
}
