use ndarray::Array1;

use super::{LU, assert_solve_ready};

impl LU {
    /// Solve a linear system with the represented matrix.
    ///
    /// This computes `x` in `A x = rhs` using the sparse LU factorization,
    /// without explicitly forming `A^{-1}`.
    pub fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.solve_initial(rhs)
    }

    pub(crate) fn solve_initial(&self, rhs: &Array1<f64>) -> Array1<f64> {
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
    fn solve_handles_generated_matrices_with_different_sparsity() {
        let mut rng = StdRng::seed_from_u64(100);
        for density in [0.0, 0.1, 0.35, 0.7, 1.0] {
            let matrix = diagonally_dominant_matrix(8, density, &mut rng);
            let expected_solution = vector(8, &mut rng);
            let rhs = matrix.dot(&expected_solution);
            let lu = LU::from_dense(matrix);

            let solution = lu.solve(&rhs);

            assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        }
    }

    #[test]
    #[should_panic(expected = "solve requires a square matrix")]
    fn solve_rejects_rectangular_matrix() {
        let matrix = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let lu = LU::from_dense(matrix);

        lu.solve(&array![1.0, 2.0]);
    }
}
