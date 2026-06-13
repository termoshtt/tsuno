use ndarray::{Array1, Array2};

use crate::lu::LU;

const BASIS_ETA_PIVOT_TOLERANCE: f64 = 1.0e-12;

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
/// This struct owns the index set $I$ as [`Basis::indices`] and a basis
/// representation that can apply $B^{-1}$ and $B^{-T}$ without explicitly
/// forming either inverse. A freshly built basis stores a sparse LU
/// factorization of $B$. Later operations may wrap that factorization with
/// product-form eta updates or with block structure for a slack row added by a
/// less-than-or-equal constraint. The type does not own the full constraint
/// matrix `A`; callers are responsible for storing `A` and passing replacement
/// columns from it.
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
/// Column replacement is stored as a product-form eta update, so callers can
/// continue solving with the updated basis without immediately rebuilding a
/// fresh sparse LU factorization.
#[derive(Debug)]
pub struct Basis {
    indices: Vec<usize>,
    representation: BasisRepresentation,
    eta_updates: Vec<BasisEtaUpdate>,
}

#[derive(Debug)]
enum BasisRepresentation {
    Factorized(LU),
    LessEqualSlack {
        base: Box<Basis>,
        basis_row: Array1<f64>,
    },
}

#[derive(Clone, Debug)]
struct BasisEtaUpdate {
    pivot: usize,
    column: Array1<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BasisUpdateError {
    RankDeficientBasis {
        rank: usize,
        dimension: usize,
    },
    PivotOutOfBounds {
        pivot: usize,
        dimension: usize,
    },
    InvalidColumnLength {
        len: usize,
        expected: usize,
    },
    SmallEtaPivot {
        pivot: usize,
        value: f64,
        tolerance: f64,
    },
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
    InvalidExtensionRowLength { len: usize, expected: usize },
    Update(BasisUpdateError),
}

impl Basis {
    /// Build `B = A_B` from the full matrix `A` and basis column indices.
    pub fn new(matrix: &Array2<f64>, indices: Vec<usize>) -> Result<Self, BasisError> {
        validate_basis_indices(matrix, &indices)?;
        let basis_matrix = basis_matrix(matrix, &indices);
        Ok(Self {
            indices,
            representation: BasisRepresentation::Factorized(LU::from_dense(basis_matrix)),
            eta_updates: Vec::new(),
        })
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            rhs.len(),
            self.dimension(),
            "right-hand side length must match the basis dimension"
        );
        let mut solution = self.solve_base(rhs);
        for eta_update in &self.eta_updates {
            eta_update.apply_inverse(&mut solution);
        }
        solution
    }

    pub fn solve_transposed(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            rhs.len(),
            self.dimension(),
            "right-hand side length must match the basis dimension"
        );
        let mut rhs = rhs.to_owned();
        for eta_update in self.eta_updates.iter().rev() {
            eta_update.apply_inverse_transposed(&mut rhs);
        }
        self.solve_base_transposed(&rhs)
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

        self.check_update_ready(position, new_column)
            .map_err(BasisError::Update)?;
        let column = self.solve(new_column);
        if column[position].abs() <= BASIS_ETA_PIVOT_TOLERANCE {
            return Err(BasisError::Update(BasisUpdateError::SmallEtaPivot {
                pivot: position,
                value: column[position],
                tolerance: BASIS_ETA_PIVOT_TOLERANCE,
            }));
        }
        self.eta_updates.push(BasisEtaUpdate {
            pivot: position,
            column,
        });
        self.indices[position] = column_index;
        Ok(())
    }

    pub fn should_refactor(&self, max_updates: usize) -> bool {
        self.update_count() >= max_updates
    }

    pub(crate) fn is_full_rank(&self) -> bool {
        self.rank() == self.dimension()
    }

    pub(crate) fn extend_with_less_equal_slack(
        self,
        slack_column: usize,
        basis_row: Array1<f64>,
    ) -> Result<Self, BasisError> {
        if basis_row.len() != self.dimension() {
            return Err(BasisError::InvalidExtensionRowLength {
                len: basis_row.len(),
                expected: self.dimension(),
            });
        }

        let mut indices = self.indices.clone();
        indices.push(slack_column);
        Ok(Self {
            indices,
            representation: BasisRepresentation::LessEqualSlack {
                base: Box::new(self),
                basis_row,
            },
            eta_updates: Vec::new(),
        })
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

    fn dimension(&self) -> usize {
        self.indices.len()
    }

    fn rank(&self) -> usize {
        self.representation.rank()
    }

    fn update_count(&self) -> usize {
        self.representation.update_count() + self.eta_updates.len()
    }

    fn solve_base(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.representation.solve(rhs)
    }

    fn solve_base_transposed(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.representation.solve_transposed(rhs)
    }

    fn check_update_ready(
        &self,
        pivot: usize,
        new_column: &Array1<f64>,
    ) -> Result<(), BasisUpdateError> {
        if self.rank() != self.dimension() {
            return Err(BasisUpdateError::RankDeficientBasis {
                rank: self.rank(),
                dimension: self.dimension(),
            });
        }
        if pivot >= self.dimension() {
            return Err(BasisUpdateError::PivotOutOfBounds {
                pivot,
                dimension: self.dimension(),
            });
        }
        if new_column.len() != self.dimension() {
            return Err(BasisUpdateError::InvalidColumnLength {
                len: new_column.len(),
                expected: self.dimension(),
            });
        }
        Ok(())
    }
}

impl BasisRepresentation {
    fn rank(&self) -> usize {
        match self {
            BasisRepresentation::Factorized(lu) => lu.row_permutation().len(),
            BasisRepresentation::LessEqualSlack { base, .. } => base.rank() + 1,
        }
    }

    fn update_count(&self) -> usize {
        match self {
            BasisRepresentation::Factorized(_) => 0,
            BasisRepresentation::LessEqualSlack { base, .. } => base.update_count(),
        }
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self {
            BasisRepresentation::Factorized(lu) => lu.solve(rhs),
            BasisRepresentation::LessEqualSlack { base, basis_row } => {
                let base_dimension = base.dimension();
                let base_rhs = Array1::from_iter(rhs.iter().take(base_dimension).copied());
                let base_solution = base.solve(&base_rhs);
                let slack_value = rhs[base_dimension] - basis_row.dot(&base_solution);
                let mut solution = Array1::zeros(base_dimension + 1);
                for (index, &value) in base_solution.iter().enumerate() {
                    solution[index] = value;
                }
                solution[base_dimension] = slack_value;
                solution
            }
        }
    }

    fn solve_transposed(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self {
            BasisRepresentation::Factorized(lu) => lu.solve_transposed(rhs),
            BasisRepresentation::LessEqualSlack { base, basis_row } => {
                let base_dimension = base.dimension();
                let slack_rhs = rhs[base_dimension];
                let base_rhs = Array1::from_iter(
                    rhs.iter()
                        .take(base_dimension)
                        .zip(basis_row.iter())
                        .map(|(&value, &row_value)| value - row_value * slack_rhs),
                );
                let base_solution = base.solve_transposed(&base_rhs);
                let mut solution = Array1::zeros(base_dimension + 1);
                for (index, &value) in base_solution.iter().enumerate() {
                    solution[index] = value;
                }
                solution[base_dimension] = slack_rhs;
                solution
            }
        }
    }
}

impl BasisEtaUpdate {
    fn apply_inverse(&self, vector: &mut Array1<f64>) {
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

    fn apply_inverse_transposed(&self, vector: &mut Array1<f64>) {
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
    fn basis_applies_multiple_column_replacements_to_solve_and_transposed_solve() {
        let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
        let first_replacement = array![7.0, 8.0, 9.0];
        let second_replacement = array![3.0, 1.0, 4.0];
        let expected_solution = array![1.0, 2.0, 5.0];
        let mut expected_basis = matrix.clone();
        let mut basis = Basis::new(&matrix, vec![0, 1, 2]).unwrap();

        basis.replace_column(1, 3, &first_replacement).unwrap();
        expected_basis.column_mut(1).assign(&first_replacement);
        basis.replace_column(0, 4, &second_replacement).unwrap();
        expected_basis.column_mut(0).assign(&second_replacement);
        let rhs = expected_basis.dot(&expected_solution);
        let transposed_rhs = expected_basis.t().dot(&expected_solution);

        let solution = basis.solve(&rhs);
        let transposed_solution = basis.solve_transposed(&transposed_rhs);

        assert_eq!(basis.indices(), &[4, 3, 2]);
        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        assert_abs_diff_eq!(transposed_solution, expected_solution, epsilon = 1.0e-9);
        assert!(basis.should_refactor(2));
    }

    #[test]
    fn basis_extends_less_equal_slack_and_solves_block_system() {
        let matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let basis = Basis::new(&matrix, vec![0, 1])
            .unwrap()
            .extend_with_less_equal_slack(2, array![2.0, 3.0])
            .unwrap();
        let expected_solution = array![4.0, 5.0, 6.0];
        let basis_matrix = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 3.0, 1.0]];
        let rhs = basis_matrix.dot(&expected_solution);

        let solution = basis.solve(&rhs);
        let transposed_rhs = basis_matrix.t().dot(&expected_solution);
        let transposed_solution = basis.solve_transposed(&transposed_rhs);

        assert_eq!(basis.indices(), &[0, 1, 2]);
        assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
        assert_abs_diff_eq!(transposed_solution, expected_solution, epsilon = 1.0e-9);
    }

    #[test]
    fn basis_replaces_column_after_less_equal_slack_extension() {
        let matrix = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let mut basis = Basis::new(&matrix, vec![0, 1])
            .unwrap()
            .extend_with_less_equal_slack(3, array![2.0, 3.0])
            .unwrap();
        let replacement = array![1.0, 1.0, 4.0];
        let expected_solution = array![2.0, 3.0, 5.0];
        let basis_matrix = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [2.0, 3.0, 4.0]];

        basis.replace_column(2, 2, &replacement).unwrap();
        let rhs = basis_matrix.dot(&expected_solution);
        let solution = basis.solve(&rhs);

        assert_eq!(basis.indices(), &[0, 1, 2]);
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
