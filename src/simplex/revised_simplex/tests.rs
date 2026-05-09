use approx::assert_abs_diff_eq;
use ndarray::array;

use super::*;
use crate::simplex::StandardFormLp;

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
fn revised_simplex_solve_returns_iteration_limit_solution() {
    let mut simplex = RevisedSimplex::with_options(
        improving_slack_lp(),
        vec![2, 3],
        RevisedSimplexOptions {
            max_iterations: 1,
            ..RevisedSimplexOptions::default()
        },
    )
    .unwrap();

    let result = simplex.solve().unwrap();

    match result {
        SimplexSolveResult::IterationLimit(solution) => {
            assert_abs_diff_eq!(
                solution.primal,
                array![0.0, 3.0, 4.0, 0.0],
                epsilon = 1.0e-9
            );
            assert_abs_diff_eq!(solution.objective_value, -6.0, epsilon = 1.0e-9);
            assert_eq!(solution.basis_indices, vec![2, 1]);
            assert_eq!(solution.iterations, 1);
        }
        _ => panic!("expected an iteration-limit solution"),
    }
}

#[test]
fn revised_simplex_can_continue_after_iteration_limit() {
    let mut simplex = RevisedSimplex::with_options(
        improving_slack_lp(),
        vec![2, 3],
        RevisedSimplexOptions {
            max_iterations: 1,
            ..RevisedSimplexOptions::default()
        },
    )
    .unwrap();

    assert!(matches!(
        simplex.solve().unwrap(),
        SimplexSolveResult::IterationLimit(_)
    ));
    assert!(matches!(
        simplex.solve().unwrap(),
        SimplexSolveResult::IterationLimit(_)
    ));
    let result = simplex.solve().unwrap();

    match result {
        SimplexSolveResult::Optimal(solution) => {
            assert_abs_diff_eq!(
                solution.primal,
                array![4.0, 3.0, 0.0, 0.0],
                epsilon = 1.0e-9
            );
            assert_eq!(solution.basis_indices, vec![0, 1]);
        }
        _ => panic!("expected continuation to reach optimality"),
    }
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
