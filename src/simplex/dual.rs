use ndarray::Array1;

use super::{
    RevisedSimplexOptions, SimplexSolution, SimplexTrace, SimplexTraceEvent, SimplexTraceStep,
};
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
/// \end{bmatrix}^T = B^{-1} b.
/// $$
///
/// Dual simplex repairs primal infeasibility by selecting a basis position
/// $p$ whose basic value is negative. The `position` field is this position
/// inside the ordered basis.
pub struct LeavingBasicVariable {
    pub position: usize,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
#[katexit::katexit]
/// Nonbasis column selected to enter the basis in a dual simplex step.
///
/// Let $p$ be the leaving basis position and let
///
/// $$
/// u = B^{-T} e_p,\qquad
/// \alpha_j = A_j^T u.
/// $$
///
/// For minimization with reduced costs $r_j \ge 0$, dual feasibility is
/// preserved by considering columns with $\alpha_j < 0$ and choosing the
/// minimum dual ratio
///
/// $$
/// \frac{r_j}{-\alpha_j}.
/// $$
pub struct EnteringColumn {
    pub column: usize,
    pub reduced_cost: f64,
    pub pivot_row_value: f64,
    pub ratio: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Optimal,
    Infeasible {
        leaving: LeavingBasicVariable,
        pivot_row: Array1<f64>,
    },
    Pivoted {
        leaving: LeavingBasicVariable,
        entering: EnteringColumn,
        pivot_row: Array1<f64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
/// Outcome of repeatedly applying dual revised simplex steps.
pub enum SolveResult {
    Optimal(SimplexSolution),
    IterationLimit(SimplexSolution),
    Infeasible {
        leaving: LeavingBasicVariable,
        pivot_row: Array1<f64>,
        iterations: usize,
    },
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
/// and then repairs negative basic primal values.
///
/// One dual revised simplex step chooses a leaving basis position
///
/// $$
/// p \in \operatorname*{argmin}_i (x_I)_i
/// \quad\text{subject to}\quad
/// (x_I)_i < -\epsilon,
/// $$
///
/// computes the pivot row
///
/// $$
/// \alpha_j = A_j^T B^{-T} e_p,
/// $$
///
/// and chooses the entering column by the dual minimum ratio test
///
/// $$
/// q \in \operatorname*{argmin}_{j:\ \alpha_j < -\epsilon}
/// \frac{r_j}{-\alpha_j}.
/// $$
///
/// If no basic value is negative, the dual-feasible basis is also primal
/// feasible and therefore optimal. If a leaving row has no eligible entering
/// column, the dual feasible dictionary proves primal infeasibility for the
/// current standard-form LP.
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

    /// Select the most infeasible basic variable.
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
    pub fn most_infeasible_basic_variable(
        &self,
    ) -> Result<Option<LeavingBasicVariable>, StandardFormError> {
        let basic_solution = self.basic_solution()?;
        Ok(most_infeasible_basic_variable(
            &basic_solution,
            self.options.pivot_tolerance,
        ))
    }

    /// Compute the dual simplex pivot row for a leaving basis position.
    ///
    /// If $p$ is the leaving basis position, define
    ///
    /// $$
    /// u = B^{-T} e_p.
    /// $$
    ///
    /// The returned vector is the full row
    ///
    /// $$
    /// \alpha = A^T u,
    /// \qquad
    /// \alpha_j = A_j^T B^{-T} e_p.
    /// $$
    ///
    /// Equivalently, $\alpha^T$ is the $p$-th row of $B^{-1} A$. For basis
    /// columns this row has the unit-vector pattern: the leaving basis column
    /// has value $1$ and the other basis columns have value $0$.
    pub fn pivot_row(
        &self,
        leaving: &LeavingBasicVariable,
    ) -> Result<Array1<f64>, StandardFormError> {
        let mut unit = Array1::zeros(self.lp.a().nrows());
        unit[leaving.position] = 1.0;
        let row_multiplier = self.basis.solve_transposed(&unit);
        Ok(self.lp.a().t().dot(&row_multiplier))
    }

    /// Select the entering column using the dual minimum ratio test.
    ///
    /// Given a pivot row $\alpha = A^T B^{-T} e_p$, a nonbasis column can enter
    /// the basis only when
    ///
    /// $$
    /// \alpha_j < -\epsilon.
    /// $$
    ///
    /// Among those candidates, this chooses
    ///
    /// $$
    /// q \in \operatorname*{argmin}_{j:\ \alpha_j < -\epsilon}
    /// \frac{r_j}{-\alpha_j},
    /// $$
    ///
    /// where $r_j = c_j - A_j^T y$ is the current reduced cost. If there is no
    /// candidate, the current dual feasible dictionary proves primal
    /// infeasibility for this leaving row.
    pub fn dual_minimum_ratio_test(
        &self,
        pivot_row: &Array1<f64>,
    ) -> Result<Option<EnteringColumn>, StandardFormError> {
        Ok(dual_minimum_ratio_test(
            self.reduced_costs()?,
            pivot_row,
            self.options.pivot_tolerance,
        ))
    }

    /// Apply one dual revised simplex step.
    ///
    /// If the basic solution already satisfies $x_I \ge -\epsilon$, the current
    /// dual feasible basis is also primal feasible and is therefore optimal for
    /// the standard-form minimization problem. Otherwise this method chooses a
    /// leaving basic variable, computes the pivot row, applies the dual ratio
    /// test, and replaces the leaving basis column with the selected entering
    /// column.
    pub fn step(&mut self) -> Result<Step, StandardFormError> {
        let Some(leaving) = self.most_infeasible_basic_variable()? else {
            return Ok(Step::Optimal);
        };

        let pivot_row = self.pivot_row(&leaving)?;
        let Some(entering) = self.dual_minimum_ratio_test(&pivot_row)? else {
            return Ok(Step::Infeasible { leaving, pivot_row });
        };

        let entering_column = self.lp.column(entering.column)?.to_owned();
        self.basis
            .replace_column(leaving.position, entering.column, &entering_column)
            .map_err(StandardFormError::Basis)?;

        Ok(Step::Pivoted {
            leaving,
            entering,
            pivot_row,
        })
    }

    /// Repeatedly apply dual revised simplex steps until termination.
    ///
    /// Each iteration applies [`DualRevisedSimplex::step`]. The method assumes
    /// the initial basis is dual feasible:
    ///
    /// $$
    /// r_j \ge -\epsilon \quad (j \notin I).
    /// $$
    ///
    /// If all basic values also satisfy
    ///
    /// $$
    /// x_I = B^{-1} b \ge -\epsilon,
    /// $$
    ///
    /// the current basis is optimal. If a negative basic value is selected but
    /// its pivot row has no entering candidate with $\alpha_j < -\epsilon$, the
    /// problem is reported as infeasible. If neither terminal condition occurs
    /// before [`RevisedSimplexOptions::max_iterations`] step attempts, this
    /// returns [`SolveResult::IterationLimit`] with the current basic solution.
    pub fn solve(
        &mut self,
        trace: &mut impl SimplexTrace,
    ) -> Result<SolveResult, StandardFormError> {
        for iteration in 0..self.options.max_iterations {
            trace.step_started(iteration, self.basis.indices());
            let step = self.step()?;
            trace.step_completed(SimplexTraceEvent {
                iteration,
                step: SimplexTraceStep::Dual(&step),
                basis_after: self.basis.indices(),
            });

            match step {
                Step::Optimal => {
                    return Ok(SolveResult::Optimal(self.current_solution(iteration)?));
                }
                Step::Infeasible { leaving, pivot_row } => {
                    return Ok(SolveResult::Infeasible {
                        leaving,
                        pivot_row,
                        iterations: iteration,
                    });
                }
                Step::Pivoted { .. } => {}
            }
        }

        Ok(SolveResult::IterationLimit(
            self.current_solution(self.options.max_iterations)?,
        ))
    }

    fn current_solution(&self, iterations: usize) -> Result<SimplexSolution, StandardFormError> {
        let basic_solution = self.basic_solution()?;
        let primal = full_primal_solution(self.lp.c().len(), self.basis.indices(), &basic_solution);
        let objective_value = self.lp.c().dot(&primal);
        Ok(SimplexSolution {
            primal,
            objective_value,
            basis_indices: self.basis.indices().to_vec(),
            iterations,
        })
    }
}

fn full_primal_solution(
    dimension: usize,
    basis_indices: &[usize],
    basic_solution: &Array1<f64>,
) -> Array1<f64> {
    let mut primal = Array1::zeros(dimension);
    for (&column, &value) in basis_indices.iter().zip(basic_solution.iter()) {
        primal[column] = value;
    }
    primal
}

fn most_infeasible_basic_variable(
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
            value: *value,
        })
        .min_by(|left, right| left.value.total_cmp(&right.value))
}

fn dual_minimum_ratio_test(
    reduced_costs: Vec<PricedColumn>,
    pivot_row: &Array1<f64>,
    tolerance: f64,
) -> Option<EnteringColumn> {
    let tolerance = tolerance.max(0.0);
    reduced_costs
        .into_iter()
        .filter_map(|priced_column| {
            let pivot_row_value = pivot_row[priced_column.column];
            (pivot_row_value < -tolerance).then(|| EnteringColumn {
                column: priced_column.column,
                reduced_cost: priced_column.reduced_cost,
                pivot_row_value,
                ratio: priced_column.reduced_cost / -pivot_row_value,
            })
        })
        .min_by(|left, right| left.ratio.total_cmp(&right.ratio))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;
    use crate::simplex::FullTrace;

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
    fn dual_revised_simplex_selects_most_infeasible_basic_variable() {
        let simplex = DualRevisedSimplex::new(two_negative_basic_values_lp(), vec![1, 2]).unwrap();

        let leaving = simplex.most_infeasible_basic_variable().unwrap();

        assert_eq!(
            leaving,
            Some(LeavingBasicVariable {
                position: 1,
                value: -3.0,
            })
        );
    }

    #[test]
    fn dual_revised_simplex_returns_none_when_basic_solution_is_primal_feasible() {
        let simplex =
            DualRevisedSimplex::new(primal_and_dual_feasible_slack_basis_lp(), vec![1, 2]).unwrap();

        let leaving = simplex.most_infeasible_basic_variable().unwrap();

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

        let leaving = simplex.most_infeasible_basic_variable().unwrap();

        assert_eq!(leaving, None);
    }

    #[test]
    fn dual_revised_simplex_computes_pivot_row() {
        let simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();
        let leaving = simplex.most_infeasible_basic_variable().unwrap().unwrap();

        let pivot_row = simplex.pivot_row(&leaving).unwrap();

        assert_abs_diff_eq!(pivot_row, array![-1.0, 1.0, 0.0], epsilon = 1.0e-9);
    }

    #[test]
    fn dual_revised_simplex_selects_entering_column_by_minimum_ratio_test() {
        let simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();
        let leaving = simplex.most_infeasible_basic_variable().unwrap().unwrap();
        let pivot_row = simplex.pivot_row(&leaving).unwrap();

        let entering = simplex
            .dual_minimum_ratio_test(&pivot_row)
            .unwrap()
            .unwrap();

        assert_eq!(
            entering,
            EnteringColumn {
                column: 0,
                reduced_cost: 1.0,
                pivot_row_value: -1.0,
                ratio: 1.0,
            }
        );
    }

    #[test]
    fn dual_revised_simplex_step_pivots_basis() {
        let mut simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();

        let step = simplex.step().unwrap();

        match step {
            Step::Pivoted {
                leaving,
                entering,
                pivot_row,
            } => {
                assert_eq!(
                    leaving,
                    LeavingBasicVariable {
                        position: 0,
                        value: -1.0,
                    }
                );
                assert_eq!(entering.column, 0);
                assert_abs_diff_eq!(pivot_row, array![-1.0, 1.0, 0.0], epsilon = 1.0e-9);
            }
            _ => panic!("expected a dual pivot"),
        }
        assert_eq!(simplex.basis().indices(), &[0, 2]);
        assert_abs_diff_eq!(
            simplex.basic_solution().unwrap(),
            array![1.0, 1.0],
            epsilon = 1.0e-9
        );
        assert!(simplex.is_dual_feasible().unwrap());
    }

    #[test]
    fn dual_revised_simplex_step_reports_optimal_when_primal_feasible() {
        let mut simplex =
            DualRevisedSimplex::new(primal_and_dual_feasible_slack_basis_lp(), vec![1, 2]).unwrap();

        let step = simplex.step().unwrap();

        assert_eq!(step, Step::Optimal);
        assert_eq!(simplex.basis().indices(), &[1, 2]);
    }

    #[test]
    fn dual_revised_simplex_step_reports_infeasible_without_entering_column() {
        let mut simplex = DualRevisedSimplex::new(
            dual_feasible_primal_infeasible_without_entering_lp(),
            vec![1, 2],
        )
        .unwrap();

        let step = simplex.step().unwrap();

        match step {
            Step::Infeasible { leaving, pivot_row } => {
                assert_eq!(
                    leaving,
                    LeavingBasicVariable {
                        position: 0,
                        value: -1.0,
                    }
                );
                assert_abs_diff_eq!(pivot_row, array![1.0, 1.0, 0.0], epsilon = 1.0e-9);
            }
            _ => panic!("expected infeasible dual step"),
        }
        assert_eq!(simplex.basis().indices(), &[1, 2]);
    }

    #[test]
    fn dual_revised_simplex_solve_returns_optimal_solution() {
        let mut simplex =
            DualRevisedSimplex::new(dual_feasible_primal_infeasible_lp(), vec![1, 2]).unwrap();
        let mut trace = FullTrace::default();

        let result = simplex.solve(&mut trace).unwrap();

        match result {
            SolveResult::Optimal(solution) => {
                assert_abs_diff_eq!(solution.primal, array![1.0, 0.0, 1.0], epsilon = 1.0e-9);
                assert_abs_diff_eq!(solution.objective_value, 1.0, epsilon = 1.0e-9);
                assert_eq!(solution.basis_indices, vec![0, 2]);
                assert_eq!(solution.iterations, 1);
            }
            _ => panic!("expected an optimal dual simplex solution"),
        }
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn dual_revised_simplex_solve_returns_iteration_limit_solution() {
        let mut simplex = DualRevisedSimplex::with_options(
            dual_feasible_primal_infeasible_lp(),
            vec![1, 2],
            RevisedSimplexOptions {
                max_iterations: 1,
                ..RevisedSimplexOptions::default()
            },
        )
        .unwrap();
        let mut trace = FullTrace::default();

        let result = simplex.solve(&mut trace).unwrap();

        match result {
            SolveResult::IterationLimit(solution) => {
                assert_abs_diff_eq!(solution.primal, array![1.0, 0.0, 1.0], epsilon = 1.0e-9);
                assert_abs_diff_eq!(solution.objective_value, 1.0, epsilon = 1.0e-9);
                assert_eq!(solution.basis_indices, vec![0, 2]);
                assert_eq!(solution.iterations, 1);
            }
            _ => panic!("expected a dual simplex iteration-limit solution"),
        }
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn dual_revised_simplex_solve_returns_infeasible_result() {
        let mut simplex = DualRevisedSimplex::new(
            dual_feasible_primal_infeasible_without_entering_lp(),
            vec![1, 2],
        )
        .unwrap();
        let mut trace = FullTrace::default();

        let result = simplex.solve(&mut trace).unwrap();

        match result {
            SolveResult::Infeasible {
                leaving,
                pivot_row,
                iterations,
            } => {
                assert_eq!(
                    leaving,
                    LeavingBasicVariable {
                        position: 0,
                        value: -1.0,
                    }
                );
                assert_abs_diff_eq!(pivot_row, array![1.0, 1.0, 0.0], epsilon = 1.0e-9);
                assert_eq!(iterations, 0);
            }
            _ => panic!("expected a dual simplex infeasible result"),
        }
        insta::assert_snapshot!(trace);
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

    fn dual_feasible_primal_infeasible_without_entering_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            array![-1.0, 2.0],
            array![1.0, 0.0, 0.0],
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
