use ndarray::{Array1, Array2, ArrayView1};

use super::{Basis, BasisError};

#[katexit::katexit]
/// Standard-form linear program for the revised simplex method.
///
/// This represents
///
/// $$
/// \min c^T x
/// \quad \text{s.t.} \quad
/// A x = b,\quad x \ge 0.
/// $$
///
/// Here
///
/// $$
/// A \in \mathbb{R}^{m \times n},\quad
/// b \in \mathbb{R}^m,\quad
/// c \in \mathbb{R}^n,\quad
/// m \le n.
/// $$
///
/// The initial implementation assumes callers provide a feasible initial
/// [`Basis`]. Phase I, which would construct such a basis automatically, is
/// intentionally outside this type for now.
///
/// Given a basis index set
///
/// $$
/// I = \{j_0, j_1, \ldots, j_{m-1}\},
/// $$
///
/// this type provides the problem-side data needed by the revised simplex
/// method. The basis cost vector is
///
/// $$
/// c_I =
/// \begin{bmatrix}
/// c_{j_0} & c_{j_1} & \cdots & c_{j_{m-1}}
/// \end{bmatrix}^T.
/// $$
///
/// The dual variables are computed from the transposed basis system
///
/// $$
/// B^T y = c_I,
/// \qquad
/// y = B^{-T} c_I.
/// $$
///
/// Then the reduced cost of column `j` is
///
/// $$
/// r_j = c_j - A_j^T y.
/// $$
///
/// These operations are exposed as [`StandardFormLp::basis_costs`],
/// [`StandardFormLp::dual_variables`], and [`StandardFormLp::reduced_cost`].
#[derive(Clone, Debug)]
pub struct StandardFormLp {
    a: Array2<f64>,
    b: Array1<f64>,
    c: Array1<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StandardFormError {
    EmptyProblem,
    TooFewColumns { nrows: usize, ncols: usize },
    RightHandSideLengthMismatch { expected: usize, actual: usize },
    CostLengthMismatch { expected: usize, actual: usize },
    DualVariableLengthMismatch { expected: usize, actual: usize },
    ColumnOutOfBounds { column: usize, ncols: usize },
    Basis(BasisError),
}

impl StandardFormLp {
    pub fn new(a: Array2<f64>, b: Array1<f64>, c: Array1<f64>) -> Result<Self, StandardFormError> {
        validate_dimensions(&a, &b, &c)?;
        Ok(Self { a, b, c })
    }

    pub fn a(&self) -> &Array2<f64> {
        &self.a
    }

    pub fn b(&self) -> &Array1<f64> {
        &self.b
    }

    pub fn c(&self) -> &Array1<f64> {
        &self.c
    }

    pub fn column(&self, column: usize) -> Result<ArrayView1<'_, f64>, StandardFormError> {
        if column >= self.a.ncols() {
            return Err(StandardFormError::ColumnOutOfBounds {
                column,
                ncols: self.a.ncols(),
            });
        }
        Ok(self.a.column(column))
    }

    pub fn basis(&self, indices: Vec<usize>) -> Result<Basis, StandardFormError> {
        Basis::new(&self.a, indices).map_err(StandardFormError::Basis)
    }

    pub fn basis_costs(&self, basis: &Basis) -> Array1<f64> {
        Array1::from_iter(basis.indices().iter().map(|&index| self.c[index]))
    }

    pub fn dual_variables(&self, basis: &Basis) -> Array1<f64> {
        let basis_costs = self.basis_costs(basis);
        basis.solve_transposed(&basis_costs)
    }

    pub fn reduced_cost(
        &self,
        dual_variables: &Array1<f64>,
        column: usize,
    ) -> Result<f64, StandardFormError> {
        if dual_variables.len() != self.a.nrows() {
            return Err(StandardFormError::DualVariableLengthMismatch {
                expected: self.a.nrows(),
                actual: dual_variables.len(),
            });
        }
        let column_view = self.column(column)?;
        Ok(self.c[column] - column_view.dot(dual_variables))
    }
}

fn validate_dimensions(
    a: &Array2<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
) -> Result<(), StandardFormError> {
    let (nrows, ncols) = a.dim();
    if nrows == 0 || ncols == 0 {
        return Err(StandardFormError::EmptyProblem);
    }
    if nrows > ncols {
        return Err(StandardFormError::TooFewColumns { nrows, ncols });
    }
    if b.len() != nrows {
        return Err(StandardFormError::RightHandSideLengthMismatch {
            expected: nrows,
            actual: b.len(),
        });
    }
    if c.len() != ncols {
        return Err(StandardFormError::CostLengthMismatch {
            expected: ncols,
            actual: c.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn standard_form_rejects_right_hand_side_length_mismatch() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![1.0];
        let c = array![1.0, 2.0];

        let error = StandardFormLp::new(a, b, c).unwrap_err();

        assert_eq!(
            error,
            StandardFormError::RightHandSideLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn standard_form_rejects_cost_length_mismatch() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![1.0, 2.0];
        let c = array![1.0];

        let error = StandardFormLp::new(a, b, c).unwrap_err();

        assert_eq!(
            error,
            StandardFormError::CostLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn column_returns_constraint_matrix_column() {
        let lp = example_lp();

        let column = lp.column(2).unwrap();

        assert_abs_diff_eq!(column, array![1.0, 0.0], epsilon = 1.0e-9);
    }

    #[test]
    fn basis_builds_from_standard_form_matrix() {
        let lp = slack_lp();

        let basis = lp.basis(vec![2, 3]).unwrap();

        assert_eq!(basis.indices(), &[2, 3]);
    }

    #[test]
    fn basis_costs_extracts_basis_cost_vector() {
        let lp = example_lp();
        let basis = lp.basis(vec![0, 1]).unwrap();

        let costs = lp.basis_costs(&basis);

        assert_abs_diff_eq!(costs, array![5.0, 4.0], epsilon = 1.0e-9);
    }

    #[test]
    fn dual_variables_solve_transposed_basis_system() {
        let lp = example_lp();
        let basis = lp.basis(vec![0, 1]).unwrap();

        let dual_variables = lp.dual_variables(&basis);

        assert_abs_diff_eq!(
            dual_variables,
            array![11.0 / 5.0, 3.0 / 5.0],
            epsilon = 1.0e-9
        );
    }

    #[test]
    fn reduced_cost_uses_dual_variables() {
        let lp = example_lp();
        let basis = lp.basis(vec![0, 1]).unwrap();
        let dual_variables = lp.dual_variables(&basis);

        let reduced_cost = lp.reduced_cost(&dual_variables, 2).unwrap();

        assert_abs_diff_eq!(reduced_cost, -6.0 / 5.0, epsilon = 1.0e-9);
    }

    #[test]
    fn reduced_cost_rejects_out_of_bounds_column() {
        let lp = example_lp();
        let y = array![0.0, 0.0];

        let error = lp.reduced_cost(&y, 3).unwrap_err();

        assert_eq!(
            error,
            StandardFormError::ColumnOutOfBounds {
                column: 3,
                ncols: 3
            }
        );
    }

    #[test]
    fn reduced_cost_rejects_dual_variable_length_mismatch() {
        let lp = example_lp();
        let y = array![0.0];

        let error = lp.reduced_cost(&y, 2).unwrap_err();

        assert_eq!(
            error,
            StandardFormError::DualVariableLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    fn slack_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn example_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[2.0, 1.0, 1.0], [1.0, 3.0, 0.0]],
            array![1.0, 1.0],
            array![5.0, 4.0, 1.0],
        )
        .unwrap()
    }
}
