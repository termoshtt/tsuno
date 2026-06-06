use ndarray::{Array1, Array2};

use crate::lu::{LU, UpdateError};

#[katexit::katexit]
/// Basis matrix representation for a standard-form revised simplex method.
///
/// For a standard-form linear program
///
/// $$
/// \min c^T x
/// \quad \text{s.t.} \quad
/// A x = b,\quad x \ge 0,
/// $$
///
/// This representation assumes that `A` has full row rank and at least as many
/// columns as rows:
///
/// $$
/// A \in \mathbb{R}^{m \times n},\quad \operatorname{rank}(A)=m,\quad m \le n.
/// $$
///
/// If `m > n`, there are not enough columns to choose an `m`-column square
/// basis matrix. Such a matrix must be reformulated, reduced by removing
/// redundant rows, or handled outside this `Basis` representation.
///
/// a simplex basis is a set `I` of `m` column indices
///
/// $$
/// I = \{j_0, j_1, \ldots, j_{m-1}\},
/// $$
///
/// where `m` is the number of rows of `A`. The corresponding basis matrix is
///
/// $$
/// B = A_I
/// = \begin{bmatrix}
/// A_{j_0} & A_{j_1} & \cdots & A_{j_{m-1}}
/// \end{bmatrix}.
/// $$
///
/// This struct owns the index set $I$ as [`Basis::indices`] and an LU
/// representation of `B`. It does not own the full constraint matrix `A`;
/// callers are responsible for storing `A` and passing replacement columns
/// from it.
///
/// In the revised simplex method, the main operations provided by this type
/// are:
///
/// $$
/// x_I = B^{-1} b,
/// $$
///
/// via [`Basis::solve`],
///
/// $$
/// y = B^{-T} c_I,
/// $$
///
/// via [`Basis::solve_transposed`], and one-column basis replacement
///
/// $$
/// B^+ =
/// \begin{bmatrix}
/// A_{j_0} & \cdots & A_q & \cdots & A_{j_{m-1}}
/// \end{bmatrix},
/// $$
///
/// via [`Basis::replace_column`].
///
/// The replacement is delegated to [`crate::lu::LU`] as a product-form eta
/// update, so callers can continue solving with the updated basis without
/// immediately rebuilding a fresh sparse LU factorization.
#[derive(Debug)]
pub struct Basis {
    indices: Vec<usize>,
    lu: LU,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BasisError {
    EmptyBasis,
    TooFewColumns { nrows: usize, ncols: usize },
    BasisSizeMismatch { expected: usize, actual: usize },
    ColumnOutOfBounds { column: usize, ncols: usize },
    CannotRemoveBasisColumn { column: usize },
    InvalidReplacementPosition { position: usize, dimension: usize },
    InvalidColumnLength { len: usize, expected: usize },
    Update(UpdateError),
}

impl Basis {
    /// Build `B = A_B` from the full matrix `A` and basis column indices.
    pub fn new(matrix: &Array2<f64>, indices: Vec<usize>) -> Result<Self, BasisError> {
        validate_basis_indices(matrix, &indices)?;
        let basis_matrix = basis_matrix(matrix, &indices);
        Ok(Self {
            indices,
            lu: LU::from_dense(basis_matrix),
        })
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn lu(&self) -> &LU {
        &self.lu
    }

    pub fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.lu.solve(rhs)
    }

    pub fn solve_transposed(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.lu.solve_transposed(rhs)
    }

    /// Replace the `position`-th basis column by `column_index`.
    ///
    /// `new_column` must be the corresponding column from the full constraint
    /// matrix. The caller owns nonbasis bookkeeping.
    pub fn replace_column(
        &mut self,
        position: usize,
        column_index: usize,
        new_column: &Array1<f64>,
    ) -> Result<(), BasisError> {
        if position >= self.indices.len() {
            return Err(BasisError::InvalidReplacementPosition {
                position,
                dimension: self.indices.len(),
            });
        }
        if new_column.len() != self.indices.len() {
            return Err(BasisError::InvalidColumnLength {
                len: new_column.len(),
                expected: self.indices.len(),
            });
        }

        self.lu
            .replace_column(position, new_column)
            .map_err(BasisError::Update)?;
        self.indices[position] = column_index;
        Ok(())
    }

    pub fn should_refactor(&self, max_updates: usize) -> bool {
        self.lu.should_refactor(max_updates)
    }

    pub(crate) fn remap_indices_after_swap_remove_column(
        mut self,
        column: usize,
        last_column: usize,
    ) -> Result<Self, BasisError> {
        if self.indices.contains(&column) {
            return Err(BasisError::CannotRemoveBasisColumn { column });
        }
        for index in &mut self.indices {
            if *index == last_column {
                *index = column;
            }
        }
        Ok(self)
    }
}

fn validate_basis_indices(matrix: &Array2<f64>, indices: &[usize]) -> Result<(), BasisError> {
    let (nrows, ncols) = matrix.dim();
    if nrows == 0 {
        return Err(BasisError::EmptyBasis);
    }
    if nrows > ncols {
        return Err(BasisError::TooFewColumns { nrows, ncols });
    }
    if indices.len() != nrows {
        return Err(BasisError::BasisSizeMismatch {
            expected: nrows,
            actual: indices.len(),
        });
    }
    for &column in indices {
        if column >= ncols {
            return Err(BasisError::ColumnOutOfBounds { column, ncols });
        }
    }
    Ok(())
}

fn basis_matrix(matrix: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let nrows = matrix.nrows();
    let mut basis = Array2::zeros((nrows, indices.len()));
    for (basis_col, &matrix_col) in indices.iter().enumerate() {
        basis
            .column_mut(basis_col)
            .assign(&matrix.column(matrix_col));
    }
    basis
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn basis_builds_matrix_and_solves() {
        let matrix = array![
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0],
            [5.0, 0.0, 0.0, 6.0],
        ];
        let basis = Basis::new(&matrix, vec![1, 2, 3]).unwrap();
        let expected_solution = array![2.0, 3.0, 5.0];
        let basis_matrix = array![[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 6.0]];
        let rhs = basis_matrix.dot(&expected_solution);

        let solution = basis.solve(&rhs);

        assert_eq!(basis.indices(), &[1, 2, 3]);
        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn basis_solves_transposed_system() {
        let matrix = array![
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0],
            [5.0, 0.0, 0.0, 6.0],
        ];
        let basis = Basis::new(&matrix, vec![1, 2, 3]).unwrap();
        let expected_solution = array![2.0, 3.0, 5.0];
        let basis_matrix = array![[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 6.0]];
        let rhs = basis_matrix.t().dot(&expected_solution);

        let solution = basis.solve_transposed(&rhs);

        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn basis_replaces_column_and_updates_indices() {
        let matrix = array![
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0],
            [5.0, 0.0, 0.0, 6.0],
        ];
        let mut basis = Basis::new(&matrix, vec![0, 2, 3]).unwrap();
        let expected_solution = array![2.0, 3.0, 5.0];
        let mut updated_basis = array![[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]];

        basis
            .replace_column(0, 1, &matrix.column(1).to_owned())
            .unwrap();
        updated_basis.column_mut(0).assign(&matrix.column(1));
        let rhs = updated_basis.dot(&expected_solution);
        let solution = basis.solve(&rhs);

        assert_eq!(basis.indices(), &[1, 2, 3]);
        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        assert!(basis.should_refactor(1));
    }

    #[test]
    fn basis_remaps_indices_after_nonbasis_column_swap_removal() {
        let matrix = array![
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0],
            [5.0, 0.0, 0.0, 6.0],
        ];
        let basis = Basis::new(&matrix, vec![1, 2, 3]).unwrap();

        let basis = basis.remap_indices_after_swap_remove_column(0, 3).unwrap();

        assert_eq!(basis.indices(), &[1, 2, 0]);
    }

    #[test]
    fn basis_rejects_removing_basis_column() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let basis = Basis::new(&matrix, vec![0, 1]).unwrap();

        let error = basis
            .remap_indices_after_swap_remove_column(1, 1)
            .unwrap_err();

        assert_eq!(error, BasisError::CannotRemoveBasisColumn { column: 1 });
    }

    #[test]
    fn basis_rejects_wrong_number_of_indices() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];

        let error = Basis::new(&matrix, vec![0]).unwrap_err();

        assert_eq!(
            error,
            BasisError::BasisSizeMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn basis_rejects_more_rows_than_columns() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let error = Basis::new(&matrix, vec![0, 1, 0]).unwrap_err();

        assert_eq!(error, BasisError::TooFewColumns { nrows: 3, ncols: 2 });
    }

    #[test]
    fn basis_rejects_out_of_bounds_column() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];

        let error = Basis::new(&matrix, vec![0, 2]).unwrap_err();

        assert_eq!(
            error,
            BasisError::ColumnOutOfBounds {
                column: 2,
                ncols: 2
            }
        );
    }
}
