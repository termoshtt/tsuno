use ndarray::Array1;

use super::RevisedSimplexOptions;
use crate::simplex::{Basis, PricedColumn, StandardFormError, StandardFormLp};

#[derive(Clone, Debug, PartialEq)]
#[katexit::katexit]
/// Basic variable selected to leave the basis in a dual simplex step.
///
/// For a basis index set $I = \{j_0,\ldots,j_{m-1}\}$, the basic values are
/// ordered as
///
/// $$
/// x_I =
/// \begin{bmatrix}
/// x_{j_0} & \cdots & x_{j_{m-1}}
/// \end{bmatrix}^T
/// =
/// B^{-1} b.
/// $$
///
/// Dual simplex repairs primal infeasibility by selecting a basis position
/// $p$ whose basic value is negative. The `position` field is this position
/// inside the ordered basis, while `column` is the corresponding original
/// column index $j_p$ in $A$.
pub struct LeavingBasicVariable {
    pub position: usize,
    pub column: usize,
    pub value: f64,
}

#[derive(Debug)]
#[katexit::katexit]
/// State for the dual revised simplex method.
///
/// This type owns the same standard-form LP data and basis representation as
/// the primal revised simplex state:
///
/// $$
/// \min c^T x
/// \quad \text{s.t.} \quad
/// A x = b,\quad x \ge 0.
/// $$
///
/// For a basis index set $I$, let $B = A_I$. The basic primal values, dual
/// variables, and nonbasis reduced costs are
///
/// $$
/// x_I = B^{-1} b,
/// \qquad
/// B^T y = c_I,
/// \qquad
/// r_j = c_j - A_j^T y \quad (j \notin I).
/// $$
///
/// Primal simplex starts from a primal feasible basis, keeps
/// $x_I \ge 0$, and repairs negative reduced costs. Dual simplex uses the
/// opposite invariant: it starts from a dual feasible basis,
///
/// $$
/// r_j \ge -\epsilon \quad (j \notin I),
/// $$
///
/// and then repairs negative basic primal values. This first implementation
/// slice exposes the shared basis-level quantities and the leaving-variable
/// selection needed to identify primal infeasibility. Pivot-row computation,
/// entering-column selection, and the dual solve loop are added separately.
pub struct DualRevisedSimplex {
    lp: StandardFormLp,
    basis: Basis,
    options: RevisedSimplexOptions,
}

impl DualRevisedSimplex {
    pub fn new(lp: StandardFormLp, basis_indices: Vec<usize>) -> Result<Self, StandardFormError> {
        Self::with_options(lp, basis_indices, RevisedSimplexOptions::default())
    }

    pub fn with_options(
        lp: StandardFormLp,
        basis_indices: Vec<usize>,
        options: RevisedSimplexOptions,
    ) -> Result<Self, StandardFormError> {
        let basis = lp.basis(basis_indices)?;
        Ok(Self { lp, basis, options })
    }

    pub fn lp(&self) -> &StandardFormLp {
        &self.lp
    }

    pub fn basis(&self) -> &Basis {
        &self.basis
    }

    pub fn options(&self) -> &RevisedSimplexOptions {
        &self.options
    }

    /// Compute the current basic primal values.
    ///
    /// This returns the vector $x_I = B^{-1} b$ in basis order. In dual
    /// simplex, these values are allowed to be negative during the search; a
    /// later dual pivot step will choose a negative component as the leaving
    /// basis position.
    pub fn basic_solution(&self) -> Result<Array1<f64>, StandardFormError> {
        self.lp.basic_solution(&self.basis)
    }

    /// Compute the dual variables for the current basis.
    ///
    /// This solves $B^T y = c_I$, equivalently $y = B^{-T} c_I$. The resulting
    /// $y$ defines reduced costs $r_j = c_j - A_j^T y$ for nonbasis columns.
    pub fn dual_variables(&self) -> Result<Array1<f64>, StandardFormError> {
        self.lp.dual_variables(&self.basis)
    }

    /// Compute reduced costs for all current nonbasis columns.
    ///
    /// For minimization, dual feasibility of the current basis means every
    /// returned reduced cost satisfies $r_j \ge -\epsilon$, where $\epsilon$ is
    /// [`RevisedSimplexOptions::reduced_cost_tolerance`].
    pub fn reduced_costs(&self) -> Result<Vec<PricedColumn>, StandardFormError> {
        self.lp.reduced_costs(&self.basis)
    }

    /// Check whether the current basis is dual feasible within tolerance.
    ///
    /// This tests
    ///
    /// $$
    /// r_j \ge -\epsilon \quad (j \notin I),
    /// $$
    ///
    /// with $\epsilon$ taken from
    /// [`RevisedSimplexOptions::reduced_cost_tolerance`]. A dual simplex
    /// implementation should only pivot from bases satisfying this condition.
    pub fn is_dual_feasible(&self) -> Result<bool, StandardFormError> {
        let tolerance = self.options.reduced_cost_tolerance.max(0.0);
        Ok(self
            .reduced_costs()?
            .iter()
            .all(|priced_column| priced_column.reduced_cost >= -tolerance))
    }

    /// Select the basic variable that should leave the basis.
    ///
    /// For dual simplex, the current basis is expected to be dual feasible, but
    /// it may be primal infeasible. Primal feasibility of the basic solution is
    ///
    /// $$
    /// x_I = B^{-1} b \ge 0.
    /// $$
    ///
    /// If every basic value satisfies $x_{I,i} \ge -\epsilon$, the current
    /// basis is primal feasible within tolerance and there is no leaving
    /// variable. Otherwise this method chooses the most negative basic value:
    ///
    /// $$
    /// p \in \operatorname*{argmin}_i (x_I)_i
    /// \quad\text{subject to}\quad
    /// (x_I)_i < -\epsilon.
    /// $$
    ///
    /// The tolerance $\epsilon$ is [`RevisedSimplexOptions::pivot_tolerance`].
    pub fn leaving_basic_variable(
        &self,
    ) -> Result<Option<LeavingBasicVariable>, StandardFormError> {
        let basic_solution = self.basic_solution()?;
        Ok(leaving_basic_variable(
            self.basis.indices(),
            &basic_solution,
            self.options.pivot_tolerance,
        ))
    }
}

fn leaving_basic_variable(
    basis_indices: &[usize],
    basic_solution: &Array1<f64>,
    tolerance: f64,
) -> Option<LeavingBasicVariable> {
    let tolerance = tolerance.max(0.0);
    basic_solution
        .iter()
        .enumerate()
        .filter(|(_, value)| **value < -tolerance)
        .map(|(position, value)| LeavingBasicVariable {
            position,
            column: basis_indices[position],
            value: *value,
        })
        .min_by(|left, right| left.value.total_cmp(&right.value))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn dual_revised_simplex_builds_basis_and_exposes_state() {
        let simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();

        assert_eq!(simplex.basis().indices(), &[1, 2]);
        assert_eq!(simplex.lp().a().nrows(), 2);
        assert_eq!(
            simplex.options().reduced_cost_tolerance,
            RevisedSimplexOptions::default().reduced_cost_tolerance
        );
    }

    #[test]
    fn dual_revised_simplex_computes_basic_solution() {
        let simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();

        let basic_solution = simplex.basic_solution().unwrap();

        assert_abs_diff_eq!(basic_solution, array![-1.0, 2.0], epsilon = 1.0e-9);
    }

    #[test]
    fn dual_revised_simplex_computes_dual_variables_and_reduced_costs() {
        let simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();

        let dual_variables = simplex.dual_variables().unwrap();
        let reduced_costs = simplex.reduced_costs().unwrap();

        assert_abs_diff_eq!(dual_variables, array![0.0, 0.0], epsilon = 1.0e-9);
        assert_eq!(
            reduced_costs,
            vec![PricedColumn {
                column: 0,
                reduced_cost: 1.0
            }]
        );
    }

    #[test]
    fn dual_revised_simplex_reports_dual_feasibility() {
        let simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();

        assert!(simplex.is_dual_feasible().unwrap());
    }

    #[test]
    fn dual_revised_simplex_reports_dual_infeasibility() {
        let simplex =
            DualRevisedSimplex::new(dual_infeasible_slack_basis_lp(), vec![1, 2]).unwrap();

        assert!(!simplex.is_dual_feasible().unwrap());
    }

    #[test]
    fn dual_revised_simplex_selects_most_negative_basic_variable() {
        let simplex = DualRevisedSimplex::new(two_negative_basic_values_lp(), vec![1, 2]).unwrap();

        let leaving = simplex.leaving_basic_variable().unwrap();

        assert_eq!(
            leaving,
            Some(LeavingBasicVariable {
                position: 1,
                column: 2,
                value: -3.0,
            })
        );
    }

    #[test]
    fn dual_revised_simplex_returns_none_when_basic_solution_is_primal_feasible() {
        let simplex =
            DualRevisedSimplex::new(primal_and_dual_feasible_slack_basis_lp(), vec![1, 2]).unwrap();

        let leaving = simplex.leaving_basic_variable().unwrap();

        assert_eq!(leaving, None);
    }

    #[test]
    fn dual_revised_simplex_leaving_selection_respects_pivot_tolerance() {
        let simplex = DualRevisedSimplex::with_options(
            nearly_primal_feasible_slack_basis_lp(),
            vec![1, 2],
            RevisedSimplexOptions {
                pivot_tolerance: 1.0e-9,
                ..RevisedSimplexOptions::default()
            },
        )
        .unwrap();

        let leaving = simplex.leaving_basic_variable().unwrap();

        assert_eq!(leaving, None);
    }

    fn dual_feasible_primal_infeasible_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            array![-1.0, 2.0],
            array![1.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn dual_infeasible_slack_basis_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            array![-1.0, 2.0],
            array![-1.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn two_negative_basic_values_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            array![-1.0, -3.0],
            array![1.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn primal_and_dual_feasible_slack_basis_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            array![1.0, 2.0],
            array![1.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn nearly_primal_feasible_slack_basis_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            array![-1.0e-10, 2.0],
            array![1.0, 0.0, 0.0],
        )
        .unwrap()
    }
}
