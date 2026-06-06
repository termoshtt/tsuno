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
/// Low-level revised simplex APIs such as
/// [`crate::simplex::primal::RevisedSimplex`] accept an explicit feasible
/// [`Basis`]. The top-level [`crate::simplex::primal::solve`] function
/// constructs such a basis with Phase I before running the Phase II primal
/// simplex method.
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
/// enter the basis. This type uses [`StandardFormLp::most_negative_reduced_cost`]
/// to pick the nonbasis column with the smallest reduced cost below a
/// caller-provided tolerance.
///
/// These operations are exposed as [`StandardFormLp::basic_solution`],
/// [`StandardFormLp::basis_costs`], [`StandardFormLp::dual_variables`],
/// [`StandardFormLp::reduced_cost`], and
/// [`StandardFormLp::most_negative_reduced_cost`].
#[derive(Clone, Debug, PartialEq)]
pub struct StandardFormLp {
    a: Array2<f64>,
    b: Array1<f64>,
    c: Array1<f64>,
}

#[katexit::katexit]
/// A column together with its reduced cost.
///
/// The `column` field is the original column index `j` in `A`, and
/// `reduced_cost` is $r_j = c_j - A_j^T y$ for that column.
#[derive(Clone, Debug, PartialEq)]
pub struct PricedColumn {
    pub column: usize,
    pub reduced_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
#[katexit::katexit]
/// Farkas infeasibility certificate for a standard-form LP.
///
/// For the infeasible system
///
/// $$
/// A x = b,\qquad x \ge 0,
/// $$
///
/// a multiplier $y$ proves infeasibility when
///
/// $$
/// A^T y \ge 0,\qquad b^T y < 0.
/// $$
///
/// Indeed, any feasible $x \ge 0$ would imply
/// $y^T A x = y^T b$, while $A^T y \ge 0$ gives
/// $y^T A x \ge 0$ and $y^T b < 0$ gives a contradiction.
///
/// # Invariant
///
/// A value of this type owns the LP it certifies and a multiplier that has
/// passed numerical certificate validation for that LP. The LP and multiplier
/// are immutable after construction, so downstream analyses such as IIS
/// extraction cannot accidentally apply the multiplier to a different problem.
pub struct FarkasCertificate {
    lp: StandardFormLp,
    multiplier: Array1<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StandardFormError {
    EmptyProblem,
    TooFewColumns {
        nrows: usize,
        ncols: usize,
    },
    RightHandSideLengthMismatch {
        expected: usize,
        actual: usize,
    },
    CostLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ColumnLengthMismatch {
        expected: usize,
        actual: usize,
    },
    RowLengthMismatch {
        expected: usize,
        actual: usize,
    },
    BasisDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    DualVariableLengthMismatch {
        expected: usize,
        actual: usize,
    },
    FarkasMultiplierLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidFarkasCertificate {
        minimum_column_value: f64,
        rhs_value: f64,
    },
    RowOutOfBounds {
        row: usize,
        nrows: usize,
    },
    ColumnOutOfBounds {
        column: usize,
        ncols: usize,
    },
    Basis(BasisError),
}

impl FarkasCertificate {
    #[katexit::katexit]
    /// Create a Farkas certificate after validating the multiplier.
    ///
    /// This returns an error unless `multiplier` has the same length as the
    /// number of rows of `lp` and satisfies
    ///
    /// $$
    /// A^T y \ge -\epsilon,\qquad b^T y < -\epsilon,
    /// $$
    ///
    /// where `tolerance` is $\epsilon$.
    pub fn new(
        lp: StandardFormLp,
        multiplier: Array1<f64>,
        tolerance: f64,
    ) -> Result<Self, StandardFormError> {
        let values = farkas_certificate_values(&lp, &multiplier)?;
        let tolerance = tolerance.max(0.0);
        if values.minimum_column_value < -tolerance || values.rhs_value >= -tolerance {
            return Err(StandardFormError::InvalidFarkasCertificate {
                minimum_column_value: values.minimum_column_value,
                rhs_value: values.rhs_value,
            });
        }
        Ok(Self { lp, multiplier })
    }

    pub fn lp(&self) -> &StandardFormLp {
        &self.lp
    }

    /// Return the verified Farkas multiplier for the stored LP.
    pub fn multiplier(&self) -> &Array1<f64> {
        &self.multiplier
    }

    #[katexit::katexit]
    /// Return the row support of the Farkas multiplier.
    ///
    /// This returns the row indices $i$ whose multiplier satisfies
    /// $|y_i| > \epsilon$, where `tolerance` is $\epsilon$. After
    /// [`FarkasCertificate::deletion_filter`], this support is the
    /// standard-form row subsystem kept by the deletion filter.
    pub fn support(&self, tolerance: f64) -> Vec<usize> {
        let tolerance = tolerance.max(0.0);
        self.multiplier
            .iter()
            .enumerate()
            .filter_map(|(row, &value)| (value.abs() > tolerance).then_some(row))
            .collect()
    }
}

struct FarkasCertificateValues {
    minimum_column_value: f64,
    rhs_value: f64,
}

fn farkas_certificate_values(
    lp: &StandardFormLp,
    multiplier: &Array1<f64>,
) -> Result<FarkasCertificateValues, StandardFormError> {
    if multiplier.len() != lp.a.nrows() {
        return Err(StandardFormError::FarkasMultiplierLengthMismatch {
            expected: lp.a.nrows(),
            actual: multiplier.len(),
        });
    }

    let column_values = lp.a.t().dot(multiplier);
    let minimum_column_value = column_values
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap();
    let rhs_value = lp.b.dot(multiplier);
    Ok(FarkasCertificateValues {
        minimum_column_value,
        rhs_value,
    })
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

    #[katexit::katexit]
    /// Replace the right-hand side vector.
    ///
    /// For
    ///
    /// $$
    /// A x = b,\qquad x \ge 0,
    /// $$
    ///
    /// this returns the LP with the same constraint matrix $A$ and objective
    /// vector $c$, but with a new right-hand side. This operation preserves the
    /// shape of the basis matrix $B = A_I$, so a caller that owns a compatible
    /// basis representation can reuse it for reoptimization.
    pub fn replace_rhs(self, b: Array1<f64>) -> Result<Self, StandardFormError> {
        Self::new(self.a, b, self.c)
    }

    #[katexit::katexit]
    /// Replace the objective cost vector.
    ///
    /// For
    ///
    /// $$
    /// \min c^T x
    /// \quad \text{s.t.} \quad
    /// A x = b,\quad x \ge 0,
    /// $$
    ///
    /// this returns the LP with the same constraint matrix $A$ and
    /// right-hand side $b$, but with a new cost vector. This operation
    /// preserves the basis matrix $B = A_I$ and the current basic values
    /// $x_I = B^{-1}b$, so a primal-feasible basis remains primal feasible for
    /// reoptimization.
    pub fn replace_cost(self, c: Array1<f64>) -> Result<Self, StandardFormError> {
        Self::new(self.a, self.b, c)
    }

    #[katexit::katexit]
    /// Replace one constraint matrix column and its objective cost.
    ///
    /// For a column index $j$, this returns the LP with $A_j$ and $c_j$
    /// replaced while all other columns, costs, and the right-hand side are
    /// unchanged. The new column must have one entry per row of $A$.
    pub fn replace_column(
        self,
        column: usize,
        values: Array1<f64>,
        cost: f64,
    ) -> Result<Self, StandardFormError> {
        if column >= self.a.ncols() {
            return Err(StandardFormError::ColumnOutOfBounds {
                column,
                ncols: self.a.ncols(),
            });
        }
        if values.len() != self.a.nrows() {
            return Err(StandardFormError::ColumnLengthMismatch {
                expected: self.a.nrows(),
                actual: values.len(),
            });
        }

        let mut a = self.a;
        let mut c = self.c;
        a.column_mut(column).assign(&values);
        c[column] = cost;
        Ok(Self { a, b: self.b, c })
    }

    #[katexit::katexit]
    /// Append one constraint matrix column and its objective cost.
    ///
    /// This returns the LP with a new last column $A_j$ and cost $c_j$, where
    /// $j$ is the returned column index. The new column must have one entry per
    /// row of $A$.
    pub fn add_column(
        self,
        values: Array1<f64>,
        cost: f64,
    ) -> Result<(Self, usize), StandardFormError> {
        let column = self.a.ncols();
        if values.len() != self.a.nrows() {
            return Err(StandardFormError::ColumnLengthMismatch {
                expected: self.a.nrows(),
                actual: values.len(),
            });
        }

        let mut a = Array2::zeros((self.a.nrows(), self.a.ncols() + 1));
        for old_column in 0..self.a.ncols() {
            a.column_mut(old_column).assign(&self.a.column(old_column));
        }
        a.column_mut(column).assign(&values);

        let mut c = Array1::zeros(self.c.len() + 1);
        for old_column in 0..self.c.len() {
            c[old_column] = self.c[old_column];
        }
        c[column] = cost;

        Ok((Self { a, b: self.b, c }, column))
    }

    #[katexit::katexit]
    /// Remove one constraint matrix column and its objective cost.
    ///
    /// This returns the LP with column $j$ removed by moving the last column
    /// into position $j$. All other column indices are unchanged. Callers that
    /// own basis indices must remap the old last column index to $j$ when the
    /// old last column is part of the basis.
    pub fn remove_column(self, column: usize) -> Result<Self, StandardFormError> {
        if column >= self.a.ncols() {
            return Err(StandardFormError::ColumnOutOfBounds {
                column,
                ncols: self.a.ncols(),
            });
        }

        let last_column = self.a.ncols() - 1;
        let mut a = Array2::zeros((self.a.nrows(), self.a.ncols() - 1));
        let mut c = Array1::zeros(self.c.len() - 1);
        for target in 0..last_column {
            let source = if target == column {
                last_column
            } else {
                target
            };
            a.column_mut(target).assign(&self.a.column(source));
            c[target] = self.c[source];
        }

        Self::new(a, self.b, c)
    }

    #[katexit::katexit]
    /// Add one less-than-or-equal constraint with a slack variable.
    ///
    /// For a row vector $a$ and upper bound $\beta$, this adds
    ///
    /// $$
    /// a^T x \le \beta
    /// $$
    ///
    /// by appending one equality row and one new slack column:
    ///
    /// $$
    /// a^T x + s = \beta,\qquad s \ge 0.
    /// $$
    ///
    /// The returned `usize` is the new slack column index. The slack cost is
    /// zero, so an optimal basis can often be reused by adding this slack
    /// column to the basis and running dual simplex only if the new slack value
    /// is negative.
    pub fn add_less_equal_constraint_with_slack(
        self,
        coefficients: Array1<f64>,
        upper_bound: f64,
    ) -> Result<(Self, usize), StandardFormError> {
        if coefficients.len() != self.a.ncols() {
            return Err(StandardFormError::RowLengthMismatch {
                expected: self.a.ncols(),
                actual: coefficients.len(),
            });
        }

        let old_rows = self.a.nrows();
        let old_cols = self.a.ncols();
        let slack_column = old_cols;
        let mut a = Array2::zeros((old_rows + 1, old_cols + 1));
        for row in 0..old_rows {
            for column in 0..old_cols {
                a[(row, column)] = self.a[(row, column)];
            }
        }
        for column in 0..old_cols {
            a[(old_rows, column)] = coefficients[column];
        }
        a[(old_rows, slack_column)] = 1.0;

        let mut b = Array1::zeros(old_rows + 1);
        for row in 0..old_rows {
            b[row] = self.b[row];
        }
        b[old_rows] = upper_bound;

        let mut c = Array1::zeros(old_cols + 1);
        for column in 0..old_cols {
            c[column] = self.c[column];
        }

        Ok((Self { a, b, c }, slack_column))
    }

    #[katexit::katexit]
    /// Return the row subsystem selected from the equality constraints.
    ///
    /// For a row index set
    ///
    /// $$
    /// R = \{i_0, i_1, \ldots, i_{k-1}\},
    /// $$
    ///
    /// this returns the standard-form LP with the same cost vector `c` and
    /// only the selected equality rows:
    ///
    /// $$
    /// A_R x = b_R,\qquad x \ge 0.
    /// $$
    ///
    /// The returned rows are ordered exactly as `rows`, so repeated row
    /// indices repeat constraints in the subsystem. `rows` must be nonempty
    /// because [`StandardFormLp`] itself does not represent empty problems.
    pub fn row_subsystem(&self, rows: &[usize]) -> Result<Self, StandardFormError> {
        let mut a = Array2::zeros((rows.len(), self.a.ncols()));
        let mut b = Array1::zeros(rows.len());
        for (target_row, &source_row) in rows.iter().enumerate() {
            if source_row >= self.a.nrows() {
                return Err(StandardFormError::RowOutOfBounds {
                    row: source_row,
                    nrows: self.a.nrows(),
                });
            }
            a.row_mut(target_row).assign(&self.a.row(source_row));
            b[target_row] = self.b[source_row];
        }
        Self::new(a, b, self.c.clone())
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

    /// Compute the current basic solution values.
    ///
    /// For a basis index set $I$ and its complement $N$, the constraint matrix
    /// is split as $A = [A_I\ A_N]$. A basic solution sets the nonbasis
    /// variables to zero:
    ///
    /// $$
    /// x_N = 0.
    /// $$
    ///
    /// Substituting this into $A x = b$ gives
    ///
    /// $$
    /// A_I x_I + A_N x_N = b,
    /// \qquad
    /// B x_I = b,
    /// $$
    ///
    /// where $B = A_I$. This returns $x_I = B^{-1} b$ in the same order as
    /// [`Basis::indices`]. Nonbasis variables are not included here and have
    /// value zero in the corresponding full basic solution.
    pub fn basic_solution(&self, basis: &Basis) -> Result<Array1<f64>, StandardFormError> {
        self.basis_column_mask(basis)?;
        Ok(basis.solve(&self.b))
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
    pub fn reduced_costs(&self, basis: &Basis) -> Result<Vec<PricedColumn>, StandardFormError> {
        let dual_variables = self.dual_variables(basis)?;
        self.nonbasis_indices(basis)?
            .into_iter()
            .map(|column| {
                self.reduced_cost(&dual_variables, column)
                    .map(|reduced_cost| PricedColumn {
                        column,
                        reduced_cost,
                    })
            })
            .collect()
    }

    /// Select the nonbasis column with the most negative reduced cost.
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
    pub fn most_negative_reduced_cost(
        &self,
        basis: &Basis,
        tolerance: f64,
    ) -> Result<Option<PricedColumn>, StandardFormError> {
        let tolerance = tolerance.max(0.0);
        Ok(self
            .reduced_costs(basis)?
            .into_iter()
            .filter(|priced_column| priced_column.reduced_cost < -tolerance)
            .min_by(|left, right| left.reduced_cost.total_cmp(&right.reduced_cost)))
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
    fn basic_solution_solves_basis_system() {
        let lp = example_lp();
        let basis = lp.basis(vec![0, 1]).unwrap();

        let basic_solution = lp.basic_solution(&basis).unwrap();

        assert_abs_diff_eq!(basic_solution, array![0.4, 0.2], epsilon = 1.0e-9);
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
    fn row_subsystem_selects_constraint_rows_in_order() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            array![1.0, 2.0, 3.0],
            array![4.0, 5.0, 6.0],
        )
        .unwrap();

        let subsystem = lp.row_subsystem(&[2, 0]).unwrap();

        assert_abs_diff_eq!(subsystem.a(), &array![[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]);
        assert_abs_diff_eq!(subsystem.b(), &array![3.0, 1.0]);
        assert_abs_diff_eq!(subsystem.c(), &array![4.0, 5.0, 6.0]);
    }

    #[test]
    fn remove_column_moves_last_column_into_removed_position() {
        let lp = StandardFormLp::new(
            array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            array![9.0, 10.0],
            array![11.0, 12.0, 13.0, 14.0],
        )
        .unwrap();

        let lp = lp.remove_column(1).unwrap();

        assert_abs_diff_eq!(lp.a(), &array![[1.0, 4.0, 3.0], [5.0, 8.0, 7.0]]);
        assert_abs_diff_eq!(lp.c(), &array![11.0, 14.0, 13.0]);
    }

    #[test]
    fn add_less_equal_constraint_with_slack_appends_row_and_slack_column() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0],
        )
        .unwrap();

        let (lp, slack_column) = lp
            .add_less_equal_constraint_with_slack(array![2.0, 3.0, 0.0], 10.0)
            .unwrap();

        assert_eq!(slack_column, 3);
        assert_abs_diff_eq!(
            lp.a(),
            &array![
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [2.0, 3.0, 0.0, 1.0],
            ]
        );
        assert_abs_diff_eq!(lp.b(), &array![4.0, 3.0, 10.0]);
        assert_abs_diff_eq!(lp.c(), &array![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn farkas_certificate_constructs_verified_standard_form_infeasibility_certificate() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [1.0, 0.0]],
            array![1.0, 2.0],
            array![0.0, 0.0],
        )
        .unwrap();
        let certificate = FarkasCertificate::new(lp.clone(), array![1.0, -1.0], 1.0e-9).unwrap();

        assert_eq!(certificate.lp(), &lp);
        assert_abs_diff_eq!(certificate.multiplier(), &array![1.0, -1.0]);
    }

    #[test]
    fn farkas_certificate_rejects_wrong_multiplier_length() {
        let lp = slack_lp();
        let error = FarkasCertificate::new(lp, array![1.0], 1.0e-9).unwrap_err();

        assert_eq!(
            error,
            StandardFormError::FarkasMultiplierLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
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
                PricedColumn {
                    column: 0,
                    reduced_cost: 1.0
                },
                PricedColumn {
                    column: 1,
                    reduced_cost: 2.0
                }
            ]
        );
    }

    #[test]
    fn most_negative_reduced_cost_selects_most_negative_reduced_cost() {
        let lp = improving_slack_lp();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let entering_column = lp.most_negative_reduced_cost(&basis, 1.0e-9).unwrap();

        assert_eq!(
            entering_column,
            Some(PricedColumn {
                column: 1,
                reduced_cost: -2.0
            })
        );
    }

    #[test]
    fn most_negative_reduced_cost_returns_none_when_reduced_costs_are_nonnegative() {
        let lp = slack_lp();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let entering_column = lp.most_negative_reduced_cost(&basis, 1.0e-9).unwrap();

        assert_eq!(entering_column, None);
    }

    #[test]
    fn most_negative_reduced_cost_respects_tolerance() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![-1.0e-8, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let basis = lp.basis(vec![2, 3]).unwrap();

        let entering_column = lp.most_negative_reduced_cost(&basis, 1.0e-7).unwrap();

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
    fn basic_solution_rejects_basis_dimension_mismatch() {
        let lp = example_lp();
        let other_matrix = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let basis = Basis::new(&other_matrix, vec![0, 1, 2]).unwrap();

        let error = lp.basic_solution(&basis).unwrap_err();

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
