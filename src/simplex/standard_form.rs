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
/// In a minimization problem, a nonbasis column with negative reduced cost can
/// enter the basis. This type uses [`StandardFormLp::entering_column`] to pick
/// the nonbasis column with the smallest reduced cost below a caller-provided
/// tolerance.
///
/// These operations are exposed as [`StandardFormLp::basis_costs`],
/// [`StandardFormLp::dual_variables`], [`StandardFormLp::reduced_cost`], and
/// [`StandardFormLp::entering_column`].
#[derive(Clone, Debug)]
pub struct StandardFormLp {
    a: Array2<f64>,
    b: Array1<f64>,
    c: Array1<f64>,
}

#[katexit::katexit]
/// Reduced cost of a single nonbasis column.
///
/// The `column` field is the original column index `j` in `A`, and `value` is
/// the reduced cost $r_j = c_j - A_j^T y$.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedCost {
    pub column: usize,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StandardFormError {
    EmptyProblem,
    TooFewColumns { nrows: usize, ncols: usize },
    RightHandSideLengthMismatch { expected: usize, actual: usize },
    CostLengthMismatch { expected: usize, actual: usize },
    BasisDimensionMismatch { expected: usize, actual: usize },
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

    /// Return the `j`-th column of the constraint matrix.
    ///
    /// The stored problem data is named `a`, `b`, and `c`, following the
    /// standard-form notation. This method returns $A_j$, the column of `A`
    /// used for pricing and basis replacement.
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

    /// Return the basis cost vector.
    ///
    /// For a basis index set $I = \{j_0, j_1, \ldots, j_{m-1}\}$, this returns
    /// $c_I = [c_{j_0}, c_{j_1}, \ldots, c_{j_{m-1}}]^T$.
    pub fn basis_costs(&self, basis: &Basis) -> Result<Array1<f64>, StandardFormError> {
        self.basis_column_mask(basis)?;
        Ok(Array1::from_iter(
            basis.indices().iter().map(|&index| self.c[index]),
        ))
    }

    /// Compute the dual variables for the given basis.
    ///
    /// For a basis matrix $B = A_I$ and basis cost vector $c_I$, this returns
    /// $y$ satisfying $B^T y = c_I$, equivalently $y = B^{-T} c_I$.
    pub fn dual_variables(&self, basis: &Basis) -> Result<Array1<f64>, StandardFormError> {
        let basis_costs = self.basis_costs(basis)?;
        Ok(basis.solve_transposed(&basis_costs))
    }

    /// Compute the reduced cost of a column.
    ///
    /// Given dual variables $y$, this returns
    /// $r_j = c_j - A_j^T y$ for the `j`-th column $A_j$.
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

    /// Return the nonbasis column indices.
    ///
    /// For the basis index set $I$, this returns the complement
    /// $\{0, 1, \ldots, n - 1\} \setminus I$ in ascending column order.
    pub fn nonbasis_indices(&self, basis: &Basis) -> Result<Vec<usize>, StandardFormError> {
        let basis_column_mask = self.basis_column_mask(basis)?;
        Ok(basis_column_mask
            .iter()
            .enumerate()
            .filter_map(|(index, &is_basis)| (!is_basis).then_some(index))
            .collect())
    }

    /// Compute reduced costs for all nonbasis columns.
    ///
    /// This first computes the dual variables $y = B^{-T} c_I$, then returns
    /// $r_j = c_j - A_j^T y$ for every $j \notin I$.
    pub fn reduced_costs(&self, basis: &Basis) -> Result<Vec<ReducedCost>, StandardFormError> {
        let dual_variables = self.dual_variables(basis)?;
        self.nonbasis_indices(basis)?
            .into_iter()
            .map(|column| {
                self.reduced_cost(&dual_variables, column)
                    .map(|value| ReducedCost { column, value })
            })
            .collect()
    }

    /// Select an entering column from the nonbasis reduced costs.
    ///
    /// With the current basis $I$, a nonbasis variable $x_j$ has value zero.
    /// If $x_j$ is increased by a small step $\theta > 0$ while preserving
    /// feasibility through the basis variables, the objective changes by
    ///
    /// $$
    /// c^T x(\theta) = c^T x(0) + \theta r_j.
    /// $$
    ///
    /// Therefore, in a minimization problem, a negative reduced cost gives a
    /// local improving direction.
    ///
    /// For this minimization problem, a nonbasis column $j \notin I$ is eligible
    /// to enter the basis when $r_j < -\epsilon$, where `tolerance` is
    /// $\epsilon$. This returns the eligible column with the smallest reduced
    /// cost, or `None` when all nonbasis reduced costs are nonnegative within
    /// the tolerance.
    pub fn entering_column(
        &self,
        basis: &Basis,
        tolerance: f64,
    ) -> Result<Option<ReducedCost>, StandardFormError> {
        let tolerance = tolerance.max(0.0);
        Ok(self
            .reduced_costs(basis)?
            .into_iter()
            .filter(|reduced_cost| reduced_cost.value < -tolerance)
            .min_by(|left, right| left.value.total_cmp(&right.value)))
    }

    fn basis_column_mask(&self, basis: &Basis) -> Result<Vec<bool>, StandardFormError> {
        if basis.indices().len() != self.a.nrows() {
            return Err(StandardFormError::BasisDimensionMismatch {
                expected: self.a.nrows(),
                actual: basis.indices().len(),
            });
        }

        let mut basis_column_mask = vec![false; self.a.ncols()];
        for &column in basis.indices() {
            if column >= self.a.ncols() {
                return Err(StandardFormError::ColumnOutOfBounds {
                    column,
                    ncols: self.a.ncols(),
                });
            }
            basis_column_mask[column] = true;
        }
        Ok(basis_column_mask)
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

        let costs = lp.basis_costs(&basis).unwrap();

        assert_abs_diff_eq!(costs, array![5.0, 4.0], epsilon = 1.0e-9);
    }

    #[test]
    fn dual_variables_solve_transposed_basis_system() {
        let lp = example_lp();
        let basis = lp.basis(vec![0, 1]).unwrap();

        let dual_variables = lp.dual_variables(&basis).unwrap();

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
        let dual_variables = lp.dual_variables(&basis).unwrap();

        let reduced_cost = lp.reduced_cost(&dual_variables, 2).unwrap();

        assert_abs_diff_eq!(reduced_cost, -6.0 / 5.0, epsilon = 1.0e-9);
    }

    #[test]
    fn nonbasis_indices_returns_basis_complement() {
        let lp = slack_lp();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let nonbasis = lp.nonbasis_indices(&basis).unwrap();

        assert_eq!(nonbasis, vec![0, 1]);
    }

    #[test]
    fn reduced_costs_returns_nonbasis_reduced_costs() {
        let lp = slack_lp();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let reduced_costs = lp.reduced_costs(&basis).unwrap();

        assert_eq!(
            reduced_costs,
            vec![
                ReducedCost {
                    column: 0,
                    value: 1.0
                },
                ReducedCost {
                    column: 1,
                    value: 2.0
                }
            ]
        );
    }

    #[test]
    fn entering_column_selects_most_negative_reduced_cost() {
        let lp = improving_slack_lp();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let entering_column = lp.entering_column(&basis, 1.0e-9).unwrap();

        assert_eq!(
            entering_column,
            Some(ReducedCost {
                column: 1,
                value: -2.0
            })
        );
    }

    #[test]
    fn entering_column_returns_none_when_reduced_costs_are_nonnegative() {
        let lp = slack_lp();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let entering_column = lp.entering_column(&basis, 1.0e-9).unwrap();

        assert_eq!(entering_column, None);
    }

    #[test]
    fn entering_column_respects_tolerance() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![-1.0e-8, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let entering_column = lp.entering_column(&basis, 1.0e-7).unwrap();

        assert_eq!(entering_column, None);
    }

    #[test]
    fn basis_costs_rejects_basis_dimension_mismatch() {
        let lp = example_lp();
        let other_matrix = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let basis = Basis::new(&other_matrix, vec![0, 1, 2]).unwrap();

        let error = lp.basis_costs(&basis).unwrap_err();

        assert_eq!(
            error,
            StandardFormError::BasisDimensionMismatch {
                expected: 2,
                actual: 3
            }
        );
    }

    #[test]
    fn nonbasis_indices_rejects_basis_column_out_of_bounds() {
        let lp = example_lp();
        let other_matrix = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let basis = Basis::new(&other_matrix, vec![0, 3]).unwrap();

        let error = lp.nonbasis_indices(&basis).unwrap_err();

        assert_eq!(
            error,
            StandardFormError::ColumnOutOfBounds {
                column: 3,
                ncols: 3
            }
        );
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

    fn improving_slack_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![-1.0, -2.0, 0.0, 0.0],
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
