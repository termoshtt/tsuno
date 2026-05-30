use ndarray::{Array1, Array2};

use super::{RevisedSimplex, SolveResult};
use crate::simplex::revised_simplex::{
    RevisedSimplexOptions, SimplexSolution, SimplexTrace, SimplexTracePhase,
};
use crate::simplex::{FarkasCertificate, StandardFormError, StandardFormLp};

#[derive(Clone, Debug, PartialEq)]
/// Infeasibility detected by the Phase I auxiliary problem.
///
/// The `certificate` field is a Farkas certificate for the original
/// standard-form LP, not for the auxiliary Phase I problem.
pub struct PhaseOneInfeasible {
    pub objective_value: f64,
    pub iterations: usize,
    pub certificate: FarkasCertificate,
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
pub(crate) enum PhaseOneResult {
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
    original_lp: StandardFormLp,
    auxiliary_lp: StandardFormLp,
    original_column_count: usize,
    initial_basis_indices: Vec<usize>,
    row_signs: Vec<f64>,
}

impl PhaseOneAuxiliaryProblem {
    pub fn new(lp: StandardFormLp) -> Self {
        let normalized = normalize_rows(&lp);
        let original_column_count = normalized.lp.a().ncols();
        let auxiliary_lp = build_auxiliary_lp(&normalized.lp);
        let initial_basis_indices =
            (original_column_count..original_column_count + normalized.lp.a().nrows()).collect();
        Self {
            original_lp: lp,
            auxiliary_lp,
            original_column_count,
            initial_basis_indices,
            row_signs: normalized.row_signs,
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
    pub(crate) fn solve(
        self,
        options: RevisedSimplexOptions,
        trace: &mut impl SimplexTrace,
    ) -> Result<PhaseOneResult, PhaseOneError> {
        let original_lp = self.original_lp;
        let original_column_count = self.original_column_count;
        let row_signs = self.row_signs;
        let mut simplex = RevisedSimplex::new(
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
            SolveResult::Optimal(solution) => {
                if solution.objective_value > options.pivot_tolerance {
                    let certificate = farkas_certificate(
                        original_lp,
                        &simplex,
                        &row_signs,
                        options.pivot_tolerance,
                    )
                    .map_err(|_| PhaseOneError::NoOriginalFeasibleBasis)?;
                    return Ok(PhaseOneResult::Infeasible(PhaseOneInfeasible {
                        objective_value: solution.objective_value,
                        iterations: solution.iterations,
                        certificate,
                    }));
                }

                // This cleanup changes the basis after Phase I optimality, but
                // it is not a simplex iteration. If traces need to explain every
                // basis representation change, add a dedicated cleanup event.
                pivot_out_artificial_columns(
                    &mut simplex,
                    original_column_count,
                    options.pivot_tolerance,
                )?;
                let basis_indices = simplex.basis().indices().to_vec();
                Ok(PhaseOneResult::Feasible { basis_indices })
            }
            SolveResult::IterationLimit(solution) => {
                Ok(PhaseOneResult::IterationLimit(PhaseOneIterationLimit {
                    auxiliary_solution: solution,
                }))
            }
            SolveResult::Unbounded { .. } => Err(PhaseOneError::NoOriginalFeasibleBasis),
        }
    }
}

fn farkas_certificate(
    original_lp: StandardFormLp,
    simplex: &RevisedSimplex,
    row_signs: &[f64],
    tolerance: f64,
) -> Result<FarkasCertificate, StandardFormError> {
    let auxiliary_dual = simplex.dual_variables()?;
    let multiplier = Array1::from_iter(
        auxiliary_dual
            .iter()
            .zip(row_signs.iter())
            .map(|(&dual_value, &row_sign)| -row_sign * dual_value),
    );
    FarkasCertificate::new(original_lp, multiplier, tolerance)
}

struct NormalizedLp {
    lp: StandardFormLp,
    row_signs: Vec<f64>,
}

fn normalize_rows(lp: &StandardFormLp) -> NormalizedLp {
    let mut a = lp.a().clone();
    let mut b = lp.b().clone();
    let mut row_signs = vec![1.0; b.len()];
    for row in 0..b.len() {
        if b[row] < 0.0 {
            row_signs[row] = -1.0;
            b[row] = -b[row];
            for column in 0..a.ncols() {
                a[[row, column]] = -a[[row, column]];
            }
        }
    }
    NormalizedLp {
        lp: StandardFormLp::new(a, b, lp.c().clone()).unwrap(),
        row_signs,
    }
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
    // Rank-deficient or redundant-row cases can leave an artificial column with
    // no original replacement. For now this is reported as a Phase I extraction
    // failure; later redundant-row handling can make this path less strict.
    let mut position = 0;
    while position < simplex.basis().indices().len() {
        if simplex.basis().indices()[position] < original_column_count {
            position += 1;
            continue;
        }

        let Some(replacement) =
            original_replacement_column(simplex, original_column_count, position, tolerance)
        else {
            return Err(PhaseOneError::NoOriginalFeasibleBasis);
        };

        simplex
            .replace_basis_column(position, replacement)
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
    let mut is_basis = vec![false; simplex.lp().a().ncols()];
    for &column in simplex.basis().indices() {
        is_basis[column] = true;
    }

    (0..original_column_count)
        .filter(|&column| !is_basis[column])
        .find(|&column| {
            let direction = simplex
                .basis_direction(column)
                .expect("candidate column should be in bounds");
            direction[position].abs() > tolerance
        })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;
    use crate::simplex::primal::solve;
    use crate::simplex::{FullTrace, SimplexResult};

    #[test]
    fn phase_one_returns_original_feasible_basis() {
        let lp = feasible_lp_without_slack_basis();
        let mut trace = FullTrace::default();

        let result = PhaseOneAuxiliaryProblem::new(lp.clone())
            .solve(RevisedSimplexOptions::default(), &mut trace)
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
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn phase_one_auxiliary_problem_builds_artificial_basis() {
        let lp = feasible_lp_without_slack_basis();

        let auxiliary = PhaseOneAuxiliaryProblem::new(lp);

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
    fn simplex_solve_records_phase_one_and_phase_two_trace() {
        let mut trace = FullTrace::default();

        let result = solve(
            feasible_lp_without_slack_basis(),
            RevisedSimplexOptions::default(),
            &mut trace,
        )
        .unwrap();

        match result {
            SimplexResult::Optimal(solution) => {
                assert_abs_diff_eq!(solution.primal, array![0.25, 0.75, 0.0], epsilon = 1.0e-9);
                assert_abs_diff_eq!(solution.objective_value, -1.0, epsilon = 1.0e-9);
            }
            _ => panic!("expected an optimal solution"),
        }
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_records_infeasible_trace() {
        let lp = infeasible_lp();
        let mut trace = FullTrace::default();

        let result = solve(lp.clone(), RevisedSimplexOptions::default(), &mut trace).unwrap();

        match result {
            SimplexResult::Infeasible(infeasible) => {
                assert_abs_diff_eq!(
                    infeasible.phase_one_objective_value.unwrap(),
                    1.0,
                    epsilon = 1.0e-9
                );
                assert_eq!(infeasible.certificate.lp(), &lp);
                let column_values = lp.a().t().dot(infeasible.certificate.multiplier());
                let minimum_column_value = column_values
                    .iter()
                    .copied()
                    .min_by(f64::total_cmp)
                    .unwrap();
                let rhs_value = lp.b().dot(infeasible.certificate.multiplier());
                assert!(minimum_column_value >= -1.0e-9);
                assert!(rhs_value < -1.0e-9);
            }
            _ => panic!("expected infeasible result"),
        }
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_returns_farkas_certificate_with_normalized_rows() {
        let lp = negative_rhs_infeasible_lp();
        let mut trace = FullTrace::default();

        let result = solve(lp.clone(), RevisedSimplexOptions::default(), &mut trace).unwrap();

        match result {
            SimplexResult::Infeasible(infeasible) => {
                assert_eq!(infeasible.certificate.lp(), &lp);
                assert_eq!(infeasible.phase_one_objective_value, Some(1.0));
                let column_values = lp.a().t().dot(infeasible.certificate.multiplier());
                let minimum_column_value = column_values
                    .iter()
                    .copied()
                    .min_by(f64::total_cmp)
                    .unwrap();
                let rhs_value = lp.b().dot(infeasible.certificate.multiplier());
                assert!(minimum_column_value >= -1.0e-9);
                assert!(rhs_value < -1.0e-9);
            }
            _ => panic!("expected infeasible result"),
        }
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_records_unbounded_trace() {
        let mut trace = FullTrace::default();

        let result = solve(unbounded_lp(), RevisedSimplexOptions::default(), &mut trace).unwrap();

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
        insta::assert_snapshot!(trace);
    }

    #[test]
    fn simplex_solve_records_phase_one_iteration_limit_trace() {
        let mut trace = FullTrace::default();

        let result = solve(
            feasible_lp_without_slack_basis(),
            RevisedSimplexOptions {
                max_iterations: 1,
                ..RevisedSimplexOptions::default()
            },
            &mut trace,
        )
        .unwrap();

        match result {
            SimplexResult::PhaseOneIterationLimit(limit) => {
                assert_eq!(limit.auxiliary_solution.basis_indices, vec![3, 0]);
                assert_abs_diff_eq!(
                    limit.auxiliary_solution.objective_value,
                    0.75,
                    epsilon = 1.0e-9
                );
            }
            _ => panic!("expected a Phase I iteration-limit result"),
        }
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
    fn simplex_solve_normalizes_negative_right_hand_side() {
        let lp = StandardFormLp::new(
            array![[-1.0, 1.0], [0.0, 1.0]],
            array![-1.0, 0.0],
            array![1.0, 0.0],
        )
        .unwrap();
        let mut trace = FullTrace::default();

        let result = solve(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();

        match result {
            SimplexResult::Optimal(solution) => {
                assert_abs_diff_eq!(solution.primal, array![1.0, 0.0], epsilon = 1.0e-9);
                assert_abs_diff_eq!(solution.objective_value, 1.0, epsilon = 1.0e-9);
            }
            _ => panic!("expected an optimal solution"),
        }
        insta::assert_snapshot!(trace);
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

    fn negative_rhs_infeasible_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[-1.0, 0.0], [1.0, 0.0]],
            array![-1.0, 2.0],
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
