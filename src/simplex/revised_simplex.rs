use std::fmt;

use ndarray::Array1;

use super::{Basis, PricedColumn, StandardFormError, StandardFormLp};

#[derive(Clone, Debug)]
pub struct RevisedSimplexOptions {
    pub reduced_cost_tolerance: f64,
    pub pivot_tolerance: f64,
    /// Maximum number of `step` calls attempted by [`RevisedSimplex::solve`].
    pub max_iterations: usize,
}

impl Default for RevisedSimplexOptions {
    fn default() -> Self {
        Self {
            reduced_cost_tolerance: 1.0e-9,
            pivot_tolerance: 1.0e-9,
            max_iterations: 1_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Original column that leaves the basis in a pivoted simplex step.
///
/// The `column` field is the original column index in `A`, not the internal
/// position inside the current ordered basis. The `step_length` field is the
/// ratio-test value at that leaving column.
pub struct LeavingColumn {
    pub column: usize,
    pub step_length: f64,
}

/// Internal leaving basis position selected by the ratio test.
#[derive(Clone, Debug, PartialEq)]
struct LeavingPosition {
    position: usize,
    step_length: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimplexStep {
    Optimal,
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
    },
    Pivoted {
        entering: PricedColumn,
        leaving: LeavingColumn,
        direction: Array1<f64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[katexit::katexit]
/// Optimal solution returned by [`RevisedSimplex::solve`].
///
/// The `primal` vector is the full decision vector `x`, including both basis
/// and nonbasis components. Nonbasis components are zero in the returned basic
/// solution:
///
/// $$
/// x_j = 0 \quad (j \notin I),
/// \qquad
/// x_I = B^{-1} b.
/// $$
///
/// The objective value is $c^T x$.
pub struct SimplexSolution {
    pub primal: Array1<f64>,
    pub objective_value: f64,
    pub basis_indices: Vec<usize>,
    pub iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
/// Outcome of repeatedly applying revised simplex steps.
pub enum SimplexSolveResult {
    Optimal(SimplexSolution),
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
        iterations: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimplexError {
    Problem(StandardFormError),
    IterationLimit { limit: usize },
}

impl From<StandardFormError> for SimplexError {
    fn from(error: StandardFormError) -> Self {
        SimplexError::Problem(error)
    }
}

pub trait SimplexTrace {
    fn step_started(&mut self, _iteration: usize, _basis: &[usize]) {}

    fn step_completed(&mut self, event: SimplexTraceEvent<'_>);
}

#[derive(Clone, Copy, Debug)]
pub struct NoTrace;

impl SimplexTrace for NoTrace {
    fn step_completed(&mut self, _event: SimplexTraceEvent<'_>) {}
}

#[derive(Clone, Debug)]
pub struct SimplexTraceEvent<'a> {
    pub iteration: usize,
    pub step: &'a SimplexStep,
    pub basis_after: &'a [usize],
}

/// Trace collector that stores every revised simplex step.
///
/// [`NoTrace`] is the zero-storage trace implementation for ordinary solves.
/// This type is the structured counterpart: it records each step so callers can
/// inspect the path taken by the simplex method, or render it via
/// [`fmt::Display`].
#[derive(Clone, Debug, Default)]
pub struct FullTrace {
    pending_basis_before: Option<Vec<usize>>,
    steps: Vec<FullTraceStep>,
}

/// One recorded revised simplex iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct FullTraceStep {
    pub iteration: usize,
    pub basis_before: Vec<usize>,
    pub outcome: FullTraceOutcome,
    pub basis_after: Vec<usize>,
}

/// Recorded outcome of one revised simplex iteration.
#[derive(Clone, Debug, PartialEq)]
pub enum FullTraceOutcome {
    Optimal,
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
    },
    Pivoted {
        entering: PricedColumn,
        leaving: LeavingColumn,
        direction: Array1<f64>,
    },
}

impl SimplexTrace for FullTrace {
    fn step_started(&mut self, _iteration: usize, basis: &[usize]) {
        self.pending_basis_before = Some(basis.to_vec());
    }

    fn step_completed(&mut self, event: SimplexTraceEvent<'_>) {
        let basis_before = self.pending_basis_before.take().unwrap();
        self.steps.push(FullTraceStep {
            iteration: event.iteration,
            basis_before,
            outcome: FullTraceOutcome::from(event.step),
            basis_after: event.basis_after.to_vec(),
        });
    }
}

impl FullTrace {
    pub fn steps(&self) -> &[FullTraceStep] {
        &self.steps
    }
}

impl From<&SimplexStep> for FullTraceOutcome {
    fn from(step: &SimplexStep) -> Self {
        match step {
            SimplexStep::Optimal => FullTraceOutcome::Optimal,
            SimplexStep::Unbounded {
                entering,
                direction,
            } => FullTraceOutcome::Unbounded {
                entering: entering.clone(),
                direction: direction.clone(),
            },
            SimplexStep::Pivoted {
                entering,
                leaving,
                direction,
            } => FullTraceOutcome::Pivoted {
                entering: entering.clone(),
                leaving: leaving.clone(),
                direction: direction.clone(),
            },
        }
    }
}

impl fmt::Display for FullTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, step) in self.steps.iter().enumerate() {
            if index > 0 {
                writeln!(f)?;
                writeln!(f)?;
            }
            write!(f, "{step}")?;
        }
        Ok(())
    }
}

impl fmt::Display for FullTraceStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "iteration {}", self.iteration)?;
        writeln!(f, "basis before: {:?}", self.basis_before)?;
        write!(f, "{}", self.outcome)?;
        write!(f, "basis after: {:?}", self.basis_after)
    }
}

impl fmt::Display for FullTraceOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FullTraceOutcome::Optimal => writeln!(f, "outcome: optimal"),
            FullTraceOutcome::Unbounded {
                entering,
                direction,
            } => {
                writeln!(f, "outcome: unbounded")?;
                writeln!(
                    f,
                    "entering column: {} (reduced_cost: {})",
                    entering.column,
                    format_number(entering.reduced_cost)
                )?;
                writeln!(f, "direction: {}", format_array(direction))
            }
            FullTraceOutcome::Pivoted {
                entering,
                leaving,
                direction,
            } => {
                writeln!(f, "outcome: pivoted")?;
                writeln!(
                    f,
                    "entering column: {} (reduced_cost: {})",
                    entering.column,
                    format_number(entering.reduced_cost)
                )?;
                writeln!(
                    f,
                    "leaving column: {} (step_length: {})",
                    leaving.column,
                    format_number(leaving.step_length)
                )?;
                writeln!(f, "direction: {}", format_array(direction))
            }
        }
    }
}

fn format_array(values: &Array1<f64>) -> String {
    let values = values
        .iter()
        .map(|&value| format_number(value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn format_number(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

#[katexit::katexit]
/// State for the revised simplex method.
///
/// This type owns the fixed standard-form problem data and the current basis
/// representation. For
///
/// $$
/// \min c^T x
/// \quad \text{s.t.} \quad
/// A x = b,\quad x \ge 0,
/// $$
///
/// the [`StandardFormLp`] stores $A$, $b$, and $c$, while the [`Basis`] stores
/// the current basis index set $I$ and an LU representation of $B = A_I$.
///
/// The methods on this type expose the quantities used by one revised simplex
/// iteration without yet performing a pivot:
///
/// $$
/// x_I = B^{-1} b,
/// \qquad
/// y = B^{-T} c_I,
/// \qquad
/// r_j = c_j - A_j^T y.
/// $$
///
/// A primal revised simplex step is organized around these quantities. First,
/// compute the current basic solution
///
/// $$
/// x_I = B^{-1} b,
/// \qquad
/// x_j = 0 \quad (j \notin I).
/// $$
///
/// Then compute the dual variables and reduced costs
///
/// $$
/// B^T y = c_I,
/// \qquad
/// r_j = c_j - A_j^T y \quad (j \notin I).
/// $$
///
/// For minimization, if every nonbasis reduced cost satisfies
///
/// $$
/// r_j \ge -\epsilon,
/// $$
///
/// the current basis is treated as optimal within tolerance. Otherwise choose
/// an entering column $q$ with negative reduced cost. The corresponding pivot
/// direction is
///
/// $$
/// d = B^{-1} A_q.
/// $$
///
/// A leaving basis position $p$ is selected by the ratio test
///
/// $$
/// p \in \operatorname*{argmin}_{i:\ d_i > \epsilon}
/// \frac{(x_I)_i}{d_i}.
/// $$
///
/// If no such $i$ exists, the problem is unbounded along the entering
/// direction. Otherwise the basis is updated by replacing the $p$-th basis
/// column with $q$.
#[derive(Debug)]
pub struct RevisedSimplex {
    lp: StandardFormLp,
    basis: Basis,
    options: RevisedSimplexOptions,
}

impl RevisedSimplex {
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

    /// Compute the current basic solution values.
    ///
    /// This is a state-level wrapper around [`StandardFormLp::basic_solution`].
    /// See that method for the mathematical definition of the returned
    /// $x_I = B^{-1} b$ vector.
    pub fn basic_solution(&self) -> Result<Array1<f64>, StandardFormError> {
        self.lp.basic_solution(&self.basis)
    }

    /// Compute the dual variables for the current basis.
    ///
    /// This is a state-level wrapper around [`StandardFormLp::dual_variables`].
    /// See that method for the transposed basis system $B^T y = c_I$.
    pub fn dual_variables(&self) -> Result<Array1<f64>, StandardFormError> {
        self.lp.dual_variables(&self.basis)
    }

    /// Compute reduced costs for all current nonbasis columns.
    ///
    /// This is a state-level wrapper around [`StandardFormLp::reduced_costs`].
    /// See that method for the definition of $r_j = c_j - A_j^T y$.
    pub fn reduced_costs(&self) -> Result<Vec<PricedColumn>, StandardFormError> {
        self.lp.reduced_costs(&self.basis)
    }

    /// Select the entering column for the current basis.
    ///
    /// This is a state-level wrapper around [`StandardFormLp::entering_column`]
    /// using [`RevisedSimplexOptions::reduced_cost_tolerance`].
    pub fn entering_column(&self) -> Result<Option<PricedColumn>, StandardFormError> {
        self.lp
            .entering_column(&self.basis, self.options.reduced_cost_tolerance)
    }

    pub fn step(&mut self) -> Result<SimplexStep, StandardFormError> {
        let Some(entering) = self.entering_column()? else {
            return Ok(SimplexStep::Optimal);
        };

        let basic_solution = self.basic_solution()?;
        let direction = self.pivot_direction(entering.column)?;
        let Some(leaving_position) =
            leaving_position(&basic_solution, &direction, self.options.pivot_tolerance)
        else {
            return Ok(SimplexStep::Unbounded {
                entering,
                direction,
            });
        };

        let leaving = LeavingColumn {
            column: self.basis.indices()[leaving_position.position],
            step_length: leaving_position.step_length,
        };

        let entering_column = self.lp.column(entering.column)?.to_owned();
        self.basis
            .replace_column(leaving_position.position, entering.column, &entering_column)
            .map_err(StandardFormError::Basis)?;

        Ok(SimplexStep::Pivoted {
            entering,
            leaving,
            direction,
        })
    }

    fn pivot_direction(&self, entering_column: usize) -> Result<Array1<f64>, StandardFormError> {
        let column = self.lp.column(entering_column)?.to_owned();
        Ok(self.basis.solve(&column))
    }

    /// Repeatedly apply revised simplex steps until termination.
    ///
    /// Each iteration applies [`RevisedSimplex::step`]. If no nonbasis column
    /// has reduced cost below the tolerance,
    ///
    /// $$
    /// r_j \ge -\epsilon \quad (j \notin I),
    /// $$
    ///
    /// the current basic solution is returned as optimal. If an entering
    /// column $q$ exists but its direction
    ///
    /// $$
    /// d = B^{-1} A_q
    /// $$
    ///
    /// has no positive component above the pivot tolerance, the problem is
    /// reported as unbounded in that direction. If neither condition occurs
    /// before [`RevisedSimplexOptions::max_iterations`] step attempts, this
    /// returns [`SimplexError::IterationLimit`].
    pub fn solve(&mut self) -> Result<SimplexSolveResult, SimplexError> {
        self.solve_with_trace(&mut NoTrace)
    }

    pub fn solve_with_trace(
        &mut self,
        trace: &mut impl SimplexTrace,
    ) -> Result<SimplexSolveResult, SimplexError> {
        for iteration in 0..self.options.max_iterations {
            trace.step_started(iteration, self.basis.indices());
            let step = self.step()?;
            trace.step_completed(SimplexTraceEvent {
                iteration,
                step: &step,
                basis_after: self.basis.indices(),
            });

            match step {
                SimplexStep::Optimal => {
                    return Ok(SimplexSolveResult::Optimal(
                        self.current_solution(iteration)?,
                    ));
                }
                SimplexStep::Unbounded {
                    entering,
                    direction,
                } => {
                    return Ok(SimplexSolveResult::Unbounded {
                        entering,
                        direction,
                        iterations: iteration,
                    });
                }
                SimplexStep::Pivoted { .. } => {}
            }
        }

        Err(SimplexError::IterationLimit {
            limit: self.options.max_iterations,
        })
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

fn leaving_position(
    basic_solution: &Array1<f64>,
    direction: &Array1<f64>,
    tolerance: f64,
) -> Option<LeavingPosition> {
    let tolerance = tolerance.max(0.0);
    basic_solution
        .iter()
        .zip(direction.iter())
        .enumerate()
        .filter(|(_, (_, direction_value))| **direction_value > tolerance)
        .map(
            |(position, (basic_value, direction_value))| LeavingPosition {
                position,
                step_length: *basic_value / *direction_value,
            },
        )
        .min_by(|left, right| left.step_length.total_cmp(&right.step_length))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn revised_simplex_builds_basis_and_computes_basic_solution() {
        let simplex = RevisedSimplex::new(example_lp(), vec![0, 1]).unwrap();

        let basic_solution = simplex.basic_solution().unwrap();

        assert_eq!(simplex.basis().indices(), &[0, 1]);
        assert_abs_diff_eq!(basic_solution, array![0.4, 0.2], epsilon = 1.0e-9);
    }

    #[test]
    fn revised_simplex_selects_entering_column_with_options() {
        let simplex = RevisedSimplex::with_options(
            improving_slack_lp(),
            vec![2, 3],
            RevisedSimplexOptions {
                reduced_cost_tolerance: 1.0e-9,
                ..RevisedSimplexOptions::default()
            },
        )
        .unwrap();

        let entering_column = simplex.entering_column().unwrap();

        assert_eq!(
            entering_column,
            Some(PricedColumn {
                column: 1,
                reduced_cost: -2.0
            })
        );
    }

    #[test]
    fn revised_simplex_respects_reduced_cost_tolerance() {
        let simplex = RevisedSimplex::with_options(
            StandardFormLp::new(
                array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
                array![4.0, 3.0],
                array![-1.0e-8, 2.0, 0.0, 0.0],
            )
            .unwrap(),
            vec![2, 3],
            RevisedSimplexOptions {
                reduced_cost_tolerance: 1.0e-7,
                ..RevisedSimplexOptions::default()
            },
        )
        .unwrap();

        let entering_column = simplex.entering_column().unwrap();

        assert_eq!(entering_column, None);
    }

    #[test]
    fn revised_simplex_step_reports_optimal_basis() {
        let mut simplex = RevisedSimplex::new(slack_lp(), vec![2, 3]).unwrap();

        let step = simplex.step().unwrap();

        assert_eq!(step, SimplexStep::Optimal);
        assert_eq!(simplex.basis().indices(), &[2, 3]);
    }

    #[test]
    fn revised_simplex_step_pivots_basis() {
        let mut simplex = RevisedSimplex::new(improving_slack_lp(), vec![2, 3]).unwrap();

        let step = simplex.step().unwrap();

        match step {
            SimplexStep::Pivoted {
                entering,
                leaving,
                direction,
            } => {
                assert_eq!(
                    entering,
                    PricedColumn {
                        column: 1,
                        reduced_cost: -2.0
                    }
                );
                assert_eq!(leaving.column, 3);
                assert_abs_diff_eq!(leaving.step_length, 3.0, epsilon = 1.0e-9);
                assert_abs_diff_eq!(direction, array![0.0, 1.0], epsilon = 1.0e-9);
            }
            _ => panic!("expected a pivoted step"),
        }
        assert_eq!(simplex.basis().indices(), &[2, 1]);
    }

    #[test]
    fn revised_simplex_step_reports_unbounded_direction() {
        let mut simplex = RevisedSimplex::new(unbounded_lp(), vec![1]).unwrap();

        let step = simplex.step().unwrap();

        match step {
            SimplexStep::Unbounded {
                entering,
                direction,
            } => {
                assert_eq!(
                    entering,
                    PricedColumn {
                        column: 0,
                        reduced_cost: -1.0
                    }
                );
                assert_abs_diff_eq!(direction, array![-1.0], epsilon = 1.0e-9);
            }
            _ => panic!("expected an unbounded step"),
        }
        assert_eq!(simplex.basis().indices(), &[1]);
    }

    #[test]
    fn revised_simplex_solve_returns_optimal_solution() {
        let mut simplex = RevisedSimplex::new(improving_slack_lp(), vec![2, 3]).unwrap();

        let result = simplex.solve().unwrap();

        match result {
            SimplexSolveResult::Optimal(solution) => {
                assert_abs_diff_eq!(
                    solution.primal,
                    array![4.0, 3.0, 0.0, 0.0],
                    epsilon = 1.0e-9
                );
                assert_abs_diff_eq!(solution.objective_value, -10.0, epsilon = 1.0e-9);
                assert_eq!(solution.basis_indices, vec![0, 1]);
                assert_eq!(solution.iterations, 2);
            }
            _ => panic!("expected an optimal solution"),
        }
    }

    #[test]
    fn revised_simplex_solve_returns_unbounded_result() {
        let mut simplex = RevisedSimplex::new(unbounded_lp(), vec![1]).unwrap();

        let result = simplex.solve().unwrap();

        match result {
            SimplexSolveResult::Unbounded {
                entering,
                direction,
                iterations,
            } => {
                assert_eq!(
                    entering,
                    PricedColumn {
                        column: 0,
                        reduced_cost: -1.0
                    }
                );
                assert_abs_diff_eq!(direction, array![-1.0], epsilon = 1.0e-9);
                assert_eq!(iterations, 0);
            }
            _ => panic!("expected an unbounded result"),
        }
    }

    #[test]
    fn revised_simplex_solve_reports_iteration_limit() {
        let mut simplex = RevisedSimplex::with_options(
            improving_slack_lp(),
            vec![2, 3],
            RevisedSimplexOptions {
                max_iterations: 1,
                ..RevisedSimplexOptions::default()
            },
        )
        .unwrap();

        let error = simplex.solve().unwrap_err();

        assert_eq!(error, SimplexError::IterationLimit { limit: 1 });
    }

    #[test]
    fn revised_simplex_solve_trace_snapshot() {
        let mut simplex = RevisedSimplex::new(improving_slack_lp(), vec![2, 3]).unwrap();
        let mut trace = FullTrace::default();

        let result = simplex.solve_with_trace(&mut trace).unwrap();

        assert!(matches!(result, SimplexSolveResult::Optimal(_)));
        insta::assert_snapshot!(trace);
    }

    fn improving_slack_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![-1.0, -2.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn slack_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn unbounded_lp() -> StandardFormLp {
        StandardFormLp::new(array![[-1.0, 1.0]], array![1.0], array![-1.0, 0.0]).unwrap()
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
