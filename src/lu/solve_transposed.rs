use ndarray::Array1;

use super::{LU, assert_solve_ready};

impl LU {
    /// Solve a transposed linear system with the represented matrix.
    ///
    /// This computes `x` in `A^T x = rhs` using the initial sparse LU
    /// factorization and accumulated eta updates, without explicitly forming
    /// `A^{-T}`.
    pub fn solve_transposed(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_solve_ready(self);
        assert_eq!(
            rhs.len(),
            self.ncols,
            "right-hand side length must match the matrix column dimension"
        );

        let mut rhs = rhs.to_owned();
        for eta_update in self.eta_updates.iter().rev() {
            eta_update.apply_inverse_transposed(&mut rhs);
        }
        self.solve_initial_transposed(&rhs)
    }

    pub(crate) fn solve_initial_transposed(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_solve_ready(self);
        assert_eq!(
            rhs.len(),
            self.ncols,
            "right-hand side length must match the matrix column dimension"
        );

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

        solution
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::super::LU;
    use super::super::test_support::{diagonally_dominant_matrix, vector};

    #[test]
    fn solve_transposed_solves_dense_rhs() {
        let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let expected_solution = array![1.0, 2.0, 5.0];
        let rhs = matrix.t().dot(&expected_solution);
        let lu = LU::from_dense(matrix);

        let solution = lu.solve_transposed(&rhs);

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn solve_transposed_handles_permuted_pivots() {
        let matrix = array![[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 6.0]];
        let expected_solution = array![3.0, 2.0, 4.0];
        let rhs = matrix.t().dot(&expected_solution);
        let lu = LU::from_dense(matrix);

        let solution = lu.solve_transposed(&rhs);

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn solve_transposed_handles_generated_matrices_with_different_sparsity() {
        let mut rng = StdRng::seed_from_u64(300);
        for density in [0.0, 0.1, 0.35, 0.7, 1.0] {
            let matrix = diagonally_dominant_matrix(8, density, &mut rng);
            let expected_solution = vector(8, &mut rng);
            let rhs = matrix.t().dot(&expected_solution);
            let lu = LU::from_dense(matrix);

            let solution = lu.solve_transposed(&rhs);

            assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        }
    }

    #[test]
    #[should_panic(expected = "solve requires a square matrix")]
    fn solve_transposed_rejects_rectangular_matrix() {
        let matrix = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let lu = LU::from_dense(matrix);

        lu.solve_transposed(&array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn solve_transposed_applies_column_replacements() {
        let mut matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let replacement = array![7.0, 8.0, 9.0];
        let expected_solution = array![1.0, 2.0, 5.0];
        let mut lu = LU::from_dense(matrix.clone());

        lu.replace_column(1, &replacement);
        matrix.column_mut(1).assign(&replacement);
        let rhs = matrix.t().dot(&expected_solution);
        let solution = lu.solve_transposed(&rhs);

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        assert_eq!(lu.update_count(), 1);
    }

    #[test]
    fn solve_transposed_applies_multiple_column_replacements() {
        let mut matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let first_replacement = array![7.0, 8.0, 9.0];
        let second_replacement = array![3.0, 1.0, 4.0];
        let expected_solution = array![1.0, 2.0, 5.0];
        let mut lu = LU::from_dense(matrix.clone());

        lu.replace_column(1, &first_replacement);
        matrix.column_mut(1).assign(&first_replacement);
        lu.replace_column(0, &second_replacement);
        matrix.column_mut(0).assign(&second_replacement);
        let rhs = matrix.t().dot(&expected_solution);
        let solution = lu.solve_transposed(&rhs);

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        assert_eq!(lu.update_count(), 2);
    }
}
