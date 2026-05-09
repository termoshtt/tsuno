use ndarray::{Array1, Array2};

use super::{
    NoTrace, RevisedSimplex, RevisedSimplexOptions, SimplexResult, SimplexSolution,
    SimplexSolveResult, SimplexTrace, SimplexTracePhase,
};
use crate::simplex::StandardFormLp;

#[derive(Clone, Debug, PartialEq)]
/// Infeasibility detected by the Phase I auxiliary problem.
pub struct PhaseOneInfeasible {
    pub objective_value: f64,
    pub iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
/// Errors that prevent extracting an original feasible basis from Phase I.
pub enum PhaseOneError {
    NoOriginalFeasibleBasis,
}

#[derive(Clone, Debug, PartialEq)]
/// Result of Phase I feasible-basis construction.
pub enum PhaseOneResult {
    Feasible { basis_indices: Vec<usize> },
    Infeasible(PhaseOneInfeasible),
    IterationLimit(SimplexSolution),
}

#[katexit::katexit]
/// Solve a standard-form LP with a Phase I feasible-basis construction.
///
/// This is the top-level primal simplex entry point for
///
/// $$
/// \min c^T x
/// \quad \text{s.t.} \quad
/// A x = b,\quad x \ge 0.
/// $$
///
/// It first solves the Phase I auxiliary problem
///
/// $$
/// \min \mathbf{1}^T w
/// \quad \text{s.t.} \quad
/// D A x + w = D b,\quad x,w \ge 0,
/// $$
///
/// where $D$ is a diagonal row-sign matrix chosen so that $D b \ge 0$.
/// If the Phase I optimum is positive, the original LP is infeasible. If it is
/// zero, the resulting feasible original basis is used to start Phase II.
///
/// The top-level solver exposes one full-featured entry point. Pass
/// [`RevisedSimplexOptions::default`] for default tolerances and iteration
/// limits, and pass [`NoTrace`] when no trace collection is needed.
pub fn solve(
    lp: StandardFormLp,
    options: RevisedSimplexOptions,
    trace: &mut impl SimplexTrace,
) -> Result<SimplexResult, super::SimplexError> {
    match find_feasible_basis_with_trace(&lp, options.clone(), trace)? {
        PhaseOneResult::Feasible { basis_indices } => {
            let mut simplex = RevisedSimplex::with_options(lp, basis_indices, options)?;
            trace.phase_started(SimplexTracePhase::PhaseTwo);
            simplex.solve_with_trace(trace).map(map_phase_two_result)
        }
        PhaseOneResult::Infeasible(infeasible) => Ok(SimplexResult::Infeasible(infeasible)),
        PhaseOneResult::IterationLimit(solution) => {
            Ok(SimplexResult::PhaseOneIterationLimit(solution))
        }
    }
}

pub fn find_feasible_basis(
    lp: &StandardFormLp,
    options: RevisedSimplexOptions,
) -> Result<PhaseOneResult, PhaseOneError> {
    find_feasible_basis_with_trace(lp, options, &mut NoTrace)
}

/// Run Phase I feasible-basis construction while recording simplex steps.
fn find_feasible_basis_with_trace(
    lp: &StandardFormLp,
    options: RevisedSimplexOptions,
    trace: &mut impl SimplexTrace,
) -> Result<PhaseOneResult, PhaseOneError> {
    let normalized = normalize_rows(lp);
    let original_column_count = lp.a().ncols();
    let auxiliary_lp = auxiliary_lp(&normalized);
    let auxiliary_basis = (original_column_count..original_column_count + lp.a().nrows()).collect();
    let mut simplex = RevisedSimplex::with_options(auxiliary_lp, auxiliary_basis, options.clone())
        .map_err(|_| PhaseOneError::NoOriginalFeasibleBasis)?;

    trace.phase_started(SimplexTracePhase::PhaseOne);
    match simplex
        .solve_with_trace(trace)
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
                original_column_count,
                options.pivot_tolerance,
            )?;
            let basis_indices = simplex.basis.indices().to_vec();
            Ok(PhaseOneResult::Feasible { basis_indices })
        }
        SimplexSolveResult::IterationLimit(solution) => {
            Ok(PhaseOneResult::IterationLimit(solution))
        }
        SimplexSolveResult::Unbounded { .. } => Err(PhaseOneError::NoOriginalFeasibleBasis),
    }
}

fn map_phase_two_result(result: SimplexSolveResult) -> SimplexResult {
    match result {
        SimplexSolveResult::Optimal(solution) => SimplexResult::Optimal(solution),
        SimplexSolveResult::IterationLimit(solution) => SimplexResult::IterationLimit(solution),
        SimplexSolveResult::Unbounded {
            entering,
            direction,
            iterations,
        } => SimplexResult::Unbounded {
            entering,
            direction,
            iterations,
        },
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

fn auxiliary_lp(lp: &StandardFormLp) -> StandardFormLp {
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
    use crate::simplex::{FullTrace, NoTrace};

    #[test]
    fn phase_one_finds_original_feasible_basis() {
        let lp = feasible_lp_without_slack_basis();

        let result = find_feasible_basis(&lp, RevisedSimplexOptions::default()).unwrap();

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
    fn simplex_solve_with_trace_records_phase_one_and_phase_two() {
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
    fn simplex_solve_reports_infeasible_from_phase_one() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [1.0, 0.0]],
            array![1.0, 2.0],
            array![0.0, 0.0],
        )
        .unwrap();

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
        let lp = StandardFormLp::new(array![[-1.0, 1.0]], array![1.0], array![-1.0, 0.0]).unwrap();

        let result = solve(lp, RevisedSimplexOptions::default(), &mut NoTrace).unwrap();

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
            SimplexResult::PhaseOneIterationLimit(solution) => {
                assert_eq!(solution.basis_indices, vec![3, 4]);
                assert_abs_diff_eq!(solution.objective_value, 1.25, epsilon = 1.0e-9);
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
}
