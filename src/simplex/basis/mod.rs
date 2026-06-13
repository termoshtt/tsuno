mod eta;
mod representation;

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use ndarray::{Array1, Array2};

use crate::lu::{LU, LuError};

use eta::BasisEtaUpdate;
use representation::BasisRepresentation;

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
    DuplicateBasisColumn { column: usize },
    RightHandSideLengthMismatch { expected: usize, actual: usize },
    CannotRemoveBasisColumn { column: usize },
    InvalidReplacementPosition { position: usize, dimension: usize },
    InvalidColumnLength { len: usize, expected: usize },
    InvalidExtensionRowLength { len: usize, expected: usize },
    Lu(LuError),
    Update(BasisUpdateError),
}

impl Basis {
    /// Build `B = A_B` from the full matrix `A` and basis column indices.
    pub fn new(matrix: &Array2<f64>, indices: Vec<usize>) -> Result<Self, BasisError> {
        validate_basis_indices(matrix, &indices)?;
        let basis_matrix = basis_matrix(matrix, &indices);
        Ok(Self {
            indices,
            representation: BasisRepresentation::Factorized(
                LU::from_dense(basis_matrix).map_err(BasisError::Lu)?,
            ),
            eta_updates: Vec::new(),
        })
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, BasisError> {
        if rhs.len() != self.dimension() {
            return Err(BasisError::RightHandSideLengthMismatch {
                expected: self.dimension(),
                actual: rhs.len(),
            });
        }
        let mut solution = self.solve_base(rhs)?;
        for eta_update in &self.eta_updates {
            eta_update.apply_inverse(&mut solution);
        }
        Ok(solution)
    }

    pub fn solve_transposed(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, BasisError> {
        if rhs.len() != self.dimension() {
            return Err(BasisError::RightHandSideLengthMismatch {
                expected: self.dimension(),
                actual: rhs.len(),
            });
        }
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
        if self
            .indices
            .iter()
            .enumerate()
            .any(|(index, &column)| index != position && column == column_index)
        {
            return Err(BasisError::DuplicateBasisColumn {
                column: column_index,
            });
        }

        self.check_update_ready(position, new_column)
            .map_err(BasisError::Update)?;
        let column = self.solve(new_column)?;
        if column[position].abs() <= BASIS_ETA_PIVOT_TOLERANCE {
            return Err(BasisError::Update(BasisUpdateError::SmallEtaPivot {
                pivot: position,
                value: column[position],
                tolerance: BASIS_ETA_PIVOT_TOLERANCE,
            }));
        }
        self.eta_updates.push(BasisEtaUpdate::new(position, column));
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

    fn solve_base(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, BasisError> {
        self.representation.solve(rhs)
    }

    fn solve_base_transposed(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, BasisError> {
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
    let mut seen = HashSet::with_capacity(indices.len());
    for &column in indices {
        if column >= ncols {
            return Err(BasisError::ColumnOutOfBounds { column, ncols });
        }
        if !seen.insert(column) {
            return Err(BasisError::DuplicateBasisColumn { column });
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
