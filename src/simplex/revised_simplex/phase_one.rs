use ndarray::{Array1, Array2};

use super::{
    RevisedSimplex, RevisedSimplexOptions, SimplexSolution, SimplexSolveResult, SimplexTrace,
    SimplexTracePhase,
};
use crate::simplex::StandardFormLp;

#[derive(Clone, Debug, PartialEq)]
/// Infeasibility detected by the Phase I auxiliary problem.
pub struct PhaseOneInfeasible {
    pub objective_value: f64,
    pub iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
/// Iteration limit reached while solving the Phase I auxiliary problem.
///
/// The contained solution belongs to the auxiliary problem, not to the original
/// LP. It may contain positive artificial variables, so it must not be treated
/// as a feasible solution of the original problem.
pub struct PhaseOneIterationLimit {
    pub auxiliary_solution: SimplexSolution,
}

#[derive(Clone, Debug, PartialEq)]
/// Errors that prevent extracting an original feasible basis from Phase I.
pub enum PhaseOneError {
    NoOriginalFeasibleBasis,
}

#[derive(Clone, Debug, PartialEq)]
/// Result of Phase I feasible-basis construction.
pub(super) enum PhaseOneResult {
    Feasible { basis_indices: Vec<usize> },
    Infeasible(PhaseOneInfeasible),
    IterationLimit(PhaseOneIterationLimit),
}

#[derive(Clone, Debug)]
#[katexit::katexit]
/// Auxiliary Phase I problem used to construct a primal feasible basis.
///
/// For an original standard-form LP
///
/// $$
/// \min c^T x
/// \quad \text{s.t.} \quad
/// A x = b,\quad x \ge 0,
/// $$
///
/// Phase I first flips rows with negative right-hand side. Let $D$ be the
/// diagonal matrix with entries $\pm 1$ chosen so that
///
/// $$
/// \bar A = D A,\qquad
/// \bar b = D b,\qquad
/// \bar b \ge 0.
/// $$
///
/// The auxiliary problem introduces one artificial variable $w_i$ per row:
///
/// $$
/// \min \mathbf{1}^T w
/// \quad \text{s.t.} \quad
/// \bar A x + I w = \bar b,\quad x,w \ge 0.
/// $$
///
/// Its constraint matrix is $[\bar A\ I]$. The artificial columns form an
/// immediate feasible basis because $w = \bar b \ge 0$. If the optimum of this
/// auxiliary problem is positive, the original LP is infeasible; if it is zero,
/// the resulting basis can be converted to an original feasible basis for
/// Phase II.
pub struct PhaseOneAuxiliaryProblem {
    auxiliary_lp: StandardFormLp,
    original_column_count: usize,
    initial_basis_indices: Vec<usize>,
}

impl PhaseOneAuxiliaryProblem {
    pub fn new(lp: &StandardFormLp) -> Self {
        let normalized = normalize_rows(lp);
        let original_column_count = normalized.a().ncols();
        let auxiliary_lp = build_auxiliary_lp(&normalized);
        let initial_basis_indices =
            (original_column_count..original_column_count + normalized.a().nrows()).collect();
        Self {
            auxiliary_lp,
            original_column_count,
            initial_basis_indices,
        }
    }

    pub fn auxiliary_lp(&self) -> &StandardFormLp {
        &self.auxiliary_lp
    }

    pub fn original_column_count(&self) -> usize {
        self.original_column_count
    }

    pub fn initial_basis_indices(&self) -> &[usize] {
        &self.initial_basis_indices
    }

    /// Solve the auxiliary Phase I problem and extract an original feasible basis.
    pub(super) fn solve(
        self,
        options: RevisedSimplexOptions,
        trace: &mut impl SimplexTrace,
    ) -> Result<PhaseOneResult, PhaseOneError> {
        let mut simplex = RevisedSimplex::with_options(
            self.auxiliary_lp,
            self.initial_basis_indices,
            options.clone(),
        )
        .map_err(|_| PhaseOneError::NoOriginalFeasibleBasis)?;

        trace.phase_started(SimplexTracePhase::PhaseOne);
        match simplex
            .solve(trace)
            .map_err(|_| PhaseOneError::NoOriginalFeasibleBasis)?
        {
            SimplexSolveResult::Optimal(solution) => {
                if solution.objective_value > options.pivot_tolerance {
                    return Ok(PhaseOneResult::Infeasible(PhaseOneInfeasible {
                        objective_value: solution.objective_value,
                        iterations: solution.iterations,
                    }));
                }

                pivot_out_artificial_columns(
                    &mut simplex,
                    self.original_column_count,
                    options.pivot_tolerance,
                )?;
                let basis_indices = simplex.basis.indices().to_vec();
                Ok(PhaseOneResult::Feasible { basis_indices })
            }
            SimplexSolveResult::IterationLimit(solution) => {
                Ok(PhaseOneResult::IterationLimit(PhaseOneIterationLimit {
                    auxiliary_solution: solution,
                }))
            }
            SimplexSolveResult::Unbounded { .. } => Err(PhaseOneError::NoOriginalFeasibleBasis),
        }
    }
}

fn normalize_rows(lp: &StandardFormLp) -> StandardFormLp {
    let mut a = lp.a().clone();
    let mut b = lp.b().clone();
    for row in 0..b.len() {
        if b[row] < 0.0 {
            b[row] = -b[row];
            for column in 0..a.ncols() {
                a[[row, column]] = -a[[row, column]];
            }
        }
    }
    StandardFormLp::new(a, b, lp.c().clone()).unwrap()
}

fn build_auxiliary_lp(lp: &StandardFormLp) -> StandardFormLp {
    let nrows = lp.a().nrows();
    let ncols = lp.a().ncols();
    let mut a = Array2::zeros((nrows, ncols + nrows));
    for row in 0..nrows {
        for column in 0..ncols {
            a[[row, column]] = lp.a()[[row, column]];
        }
        a[[row, ncols + row]] = 1.0;
    }

    let mut c = Array1::zeros(ncols + nrows);
    for artificial in ncols..ncols + nrows {
        c[artificial] = 1.0;
    }

    StandardFormLp::new(a, lp.b().clone(), c).unwrap()
}

fn pivot_out_artificial_columns(
    simplex: &mut RevisedSimplex,
    original_column_count: usize,
    tolerance: f64,
) -> Result<(), PhaseOneError> {
    let mut position = 0;
    while position < simplex.basis.indices().len() {
        if simplex.basis.indices()[position] < original_column_count {
            position += 1;
            continue;
        }

        let Some(replacement) =
            original_replacement_column(simplex, original_column_count, position, tolerance)
        else {
            return Err(PhaseOneError::NoOriginalFeasibleBasis);
        };

        let column = simplex
            .lp
            .column(replacement)
            .map_err(|_| PhaseOneError::NoOriginalFeasibleBasis)?
            .to_owned();
        simplex
            .basis
            .replace_column(position, replacement, &column)
            .map_err(|_| PhaseOneError::NoOriginalFeasibleBasis)?;
        position += 1;
    }
    Ok(())
}

fn original_replacement_column(
    simplex: &RevisedSimplex,
    original_column_count: usize,
    position: usize,
    tolerance: f64,
) -> Option<usize> {
    let mut is_basis = vec![false; simplex.lp.a().ncols()];
    for &column in simplex.basis.indices() {
        is_basis[column] = true;
    }

    (0..original_column_count)
        .filter(|&column| !is_basis[column])
        .find(|&column| {
            let candidate = simplex.lp.column(column).unwrap().to_owned();
            let direction = simplex.basis.solve(&candidate);
            direction[position].abs() > tolerance
        })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;
    use crate::simplex::{FullTrace, NoTrace, SimplexResult, solve};

    #[test]
    fn phase_one_returns_original_feasible_basis() {
        let lp = feasible_lp_without_slack_basis();

        let result = PhaseOneAuxiliaryProblem::new(&lp)
            .solve(RevisedSimplexOptions::default(), &mut NoTrace)
            .unwrap();

        match result {
            PhaseOneResult::Feasible { basis_indices } => {
                assert!(basis_indices.iter().all(|&column| column < lp.a().ncols()));
                let basis = lp.basis(basis_indices).unwrap();
                let basic_solution = lp.basic_solution(&basis).unwrap();
                assert!(basic_solution.iter().all(|&value| value >= -1.0e-9));
            }
            _ => panic!("expected a feasible basis"),
        }
    }

    #[test]
    fn phase_one_auxiliary_problem_builds_artificial_basis() {
        let lp = feasible_lp_without_slack_basis();

        let auxiliary = PhaseOneAuxiliaryProblem::new(&lp);

        assert_eq!(auxiliary.original_column_count(), 3);
        assert_eq!(auxiliary.initial_basis_indices(), &[3, 4]);
        assert_abs_diff_eq!(
            auxiliary.auxiliary_lp().a(),
            &array![[1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]],
            epsilon = 1.0e-9
        );
        assert_abs_diff_eq!(
            auxiliary.auxiliary_lp().b(),
            &array![1.0, 0.25],
            epsilon = 1.0e-9
        );
        assert_abs_diff_eq!(
            auxiliary.auxiliary_lp().c(),
            &array![0.0, 0.0, 0.0, 1.0, 1.0],
            epsilon = 1.0e-9
        );
    }

    #[test]
    fn simplex_solve_uses_phase_one_basis_for_phase_two() {
        let result = solve(
            feasible_lp_without_slack_basis(),
            RevisedSimplexOptions::default(),
            &mut NoTrace,
        )
        .unwrap();

        match result {
            SimplexResult::Optimal(solution) => {
                assert_abs_diff_eq!(solution.primal, array![0.25, 0.75, 0.0], epsilon = 1.0e-9);
                assert_abs_diff_eq!(solution.objective_value, -1.0, epsilon = 1.0e-9);
            }
            _ => panic!("expected an optimal solution"),
        }
    }

    #[test]
    fn simplex_solve_records_phase_one_and_phase_two_trace() {
        let mut trace = FullTrace::default();

        let result = solve(
            feasible_lp_without_slack_basis(),
            RevisedSimplexOptions::default(),
            &mut trace,
        )
        .unwrap();

        assert!(matches!(result, SimplexResult::Optimal(_)));
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_records_infeasible_trace() {
        let lp = infeasible_lp();
        let mut trace = FullTrace::default();

        let result = solve(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();

        assert!(matches!(result, SimplexResult::Infeasible(_)));
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_records_unbounded_trace() {
        let mut trace = FullTrace::default();

        let result = solve(unbounded_lp(), RevisedSimplexOptions::default(), &mut trace).unwrap();

        assert!(matches!(result, SimplexResult::Unbounded { .. }));
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_records_phase_one_iteration_limit_trace() {
        let mut trace = FullTrace::default();

        let result = solve(
            feasible_lp_without_slack_basis(),
            RevisedSimplexOptions {
                max_iterations: 0,
                ..RevisedSimplexOptions::default()
            },
            &mut trace,
        )
        .unwrap();

        assert!(matches!(result, SimplexResult::PhaseOneIterationLimit(_)));
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_records_phase_two_iteration_limit_trace() {
        let mut trace = FullTrace::default();

        let result = solve(
            phase_two_iteration_limit_lp(),
            RevisedSimplexOptions {
                max_iterations: 1,
                ..RevisedSimplexOptions::default()
            },
            &mut trace,
        )
        .unwrap();

        assert!(
            matches!(result, SimplexResult::IterationLimit(_)),
            "expected Phase II iteration limit, got {result:?}"
        );
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_reports_infeasible_from_phase_one() {
        let lp = infeasible_lp();

        let result = solve(lp, RevisedSimplexOptions::default(), &mut NoTrace).unwrap();

        match result {
            SimplexResult::Infeasible(infeasible) => {
                assert_abs_diff_eq!(infeasible.objective_value, 1.0, epsilon = 1.0e-9);
            }
            _ => panic!("expected infeasible result"),
        }
    }

    #[test]
    fn simplex_solve_reports_phase_two_unbounded() {
        let result = solve(
            unbounded_lp(),
            RevisedSimplexOptions::default(),
            &mut NoTrace,
        )
        .unwrap();

        match result {
            SimplexResult::Unbounded {
                entering,
                direction,
                ..
            } => {
                assert_eq!(entering.column, 0);
                assert_abs_diff_eq!(direction, array![-1.0], epsilon = 1.0e-9);
            }
            _ => panic!("expected an unbounded result"),
        }
    }

    #[test]
    fn simplex_solve_normalizes_negative_right_hand_side() {
        let lp = StandardFormLp::new(
            array![[-1.0, 1.0], [0.0, 1.0]],
            array![-1.0, 0.0],
            array![1.0, 0.0],
        )
        .unwrap();

        let result = solve(lp, RevisedSimplexOptions::default(), &mut NoTrace).unwrap();

        match result {
            SimplexResult::Optimal(solution) => {
                assert_abs_diff_eq!(solution.primal, array![1.0, 0.0], epsilon = 1.0e-9);
                assert_abs_diff_eq!(solution.objective_value, 1.0, epsilon = 1.0e-9);
            }
            _ => panic!("expected an optimal solution"),
        }
    }

    #[test]
    fn simplex_solve_reports_phase_one_iteration_limit() {
        let result = solve(
            feasible_lp_without_slack_basis(),
            RevisedSimplexOptions {
                max_iterations: 0,
                ..RevisedSimplexOptions::default()
            },
            &mut NoTrace,
        )
        .unwrap();

        match result {
            SimplexResult::PhaseOneIterationLimit(limit) => {
                assert_eq!(limit.auxiliary_solution.basis_indices, vec![3, 4]);
                assert_abs_diff_eq!(
                    limit.auxiliary_solution.objective_value,
                    1.25,
                    epsilon = 1.0e-9
                );
            }
            _ => panic!("expected a Phase I iteration-limit result"),
        }
    }

    fn feasible_lp_without_slack_basis() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            array![1.0, 0.25],
            array![-1.0, -1.0, 0.0],
        )
        .unwrap()
    }

    fn infeasible_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 0.0], [1.0, 0.0]],
            array![1.0, 2.0],
            array![0.0, 0.0],
        )
        .unwrap()
    }

    fn unbounded_lp() -> StandardFormLp {
        StandardFormLp::new(array![[-1.0, 1.0]], array![1.0], array![-1.0, 0.0]).unwrap()
    }

    fn phase_two_iteration_limit_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[-1.0, -2.0, -3.0]],
            array![0.0],
            array![3.0, 0.0, 1.0],
        )
        .unwrap()
    }
}
