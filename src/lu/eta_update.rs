use ndarray::Array1;

use super::{LU, assert_solve_ready};

/// Product-form column replacement update.
///
/// If the `pivot`-th column of the represented matrix `A` is replaced by
/// `new_column`, and `column = A^{-1} new_column`, the updated matrix is
/// `A E`, where `E` is the identity matrix with column `pivot` replaced by
/// `column`.
#[derive(Clone, Debug)]
pub struct EtaUpdate {
    pivot: usize,
    column: Array1<f64>,
}

impl EtaUpdate {
    pub fn pivot(&self) -> usize {
        self.pivot
    }

    pub fn column(&self) -> &Array1<f64> {
        &self.column
    }

    pub(crate) fn apply_inverse(&self, vector: &mut Array1<f64>) {
        debug_assert_eq!(vector.len(), self.column.len());

        let pivot_value = self.column[self.pivot];
        let pivot = vector[self.pivot] / pivot_value;
        for index in 0..vector.len() {
            if index != self.pivot {
                vector[index] -= self.column[index] * pivot;
            }
        }
        vector[self.pivot] = pivot;
    }

    pub(crate) fn apply_inverse_transposed(&self, vector: &mut Array1<f64>) {
        debug_assert_eq!(vector.len(), self.column.len());

        let pivot_value = self.column[self.pivot];
        let off_pivot_dot = self
            .column
            .iter()
            .zip(vector.iter())
            .enumerate()
            .filter(|(index, _)| *index != self.pivot)
            .map(|(_, (column, value))| column * value)
            .sum::<f64>();
        vector[self.pivot] = (vector[self.pivot] - off_pivot_dot) / pivot_value;
    }
}

impl LU {
    /// Replace one column of the represented matrix by storing an eta update.
    ///
    /// This updates the represented matrix from `A` to `A E`, where `E` is the
    /// identity matrix with `pivot` column replaced by `A^{-1} new_column`.
    /// The underlying sparse LU factors are not recomputed.
    pub fn replace_column(&mut self, pivot: usize, new_column: &Array1<f64>) {
        assert_solve_ready(self);
        assert!(
            pivot < self.ncols,
            "replacement column index must be in bounds"
        );
        assert_eq!(
            new_column.len(),
            self.nrows,
            "replacement column length must match the matrix row dimension"
        );

        let column = self.solve(new_column);
        assert!(column[pivot] != 0.0, "eta pivot entry must be non-zero");
        self.eta_updates.push(EtaUpdate { pivot, column });
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::super::LU;

    #[test]
    fn replace_column_reconstructs_updated_matrix() {
        let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let replacement = array![7.0, 8.0, 9.0];
        let mut expected = matrix.clone();
        let mut lu = LU::from_dense(matrix);

        lu.replace_column(1, &replacement);
        expected.column_mut(1).assign(&replacement);

        assert_abs_diff_eq!(lu.reconstruct(), expected, epsilon = 1.0e-9);
    }

    #[test]
    fn replace_column_accumulates_multiple_updates() {
        let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let first_replacement = array![7.0, 8.0, 9.0];
        let second_replacement = array![3.0, 1.0, 4.0];
        let mut expected = matrix.clone();
        let mut lu = LU::from_dense(matrix);

        lu.replace_column(1, &first_replacement);
        expected.column_mut(1).assign(&first_replacement);
        lu.replace_column(0, &second_replacement);
        expected.column_mut(0).assign(&second_replacement);

        assert_abs_diff_eq!(lu.reconstruct(), expected, epsilon = 1.0e-9);
        assert_eq!(lu.update_count(), 2);
        assert_eq!(lu.eta_updates()[0].pivot(), 1);
        assert_eq!(lu.eta_updates()[1].pivot(), 0);
    }

    #[test]
    #[should_panic(expected = "replacement column index must be in bounds")]
    fn replace_column_rejects_out_of_bounds_pivot() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let mut lu = LU::from_dense(matrix);

        lu.replace_column(2, &array![1.0, 2.0]);
    }
}
