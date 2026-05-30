use ndarray::Array1;

use crate::simplex::dual::{
    DualRevisedSimplex, DualSimplexError, LeavingBasicVariable, SolveResult as DualSolveResult,
};
use crate::simplex::primal::{
    PrimalSimplexError, RevisedSimplex, SolveResult as PrimalSolveResult,
    primal_infeasible_basic_value,
};
use crate::simplex::{
    FarkasCertificate, PricedColumn, RevisedSimplexOptions, RevisedSimplexState, SimplexError,
    SimplexSolution, SimplexTrace, StandardFormError, StandardFormLp,
};

#[derive(Debug)]
/// Solver state selected for warm-start reoptimization.
///
/// This is the first reoptimization entry point for modified LPs. It rebuilds
/// the basis representation from a caller-provided basis index set and chooses
/// the simplex variant whose invariant is immediately satisfied.
pub enum WarmStart {
    /// The reused basis is primal feasible, so primal revised simplex can
    /// continue from it.
    Primal(RevisedSimplex),
    /// The reused basis is not primal feasible but is dual feasible, so dual
    /// revised simplex can repair primal infeasibility from it.
    Dual(DualRevisedSimplex),
}

#[derive(Clone, Debug, PartialEq)]
#[katexit::katexit]
/// Outcome of solving a warm-started standard-form LP.
///
/// Warm start chooses either primal or dual revised simplex from the reused
/// basis invariant and then delegates to that solver. Therefore this result is
/// the union of the terminal outcomes reachable from those already existing
/// solve loops:
///
/// - primal revised simplex can prove optimality, hit an iteration limit, or
///   prove primal unboundedness;
/// - dual revised simplex can prove optimality, hit an iteration limit, or
///   prove primal infeasibility with a Farkas certificate.
///
/// No Phase I outcome appears here because the caller has already supplied a
/// reusable basis.
pub enum WarmStartResult {
    Optimal(SimplexSolution),
    IterationLimit(SimplexSolution),
    /// The primal standard-form LP is infeasible.
    Infeasible {
        leaving: LeavingBasicVariable,
        pivot_row: Array1<f64>,
        certificate: FarkasCertificate,
        iterations: usize,
    },
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
        iterations: usize,
    },
}

impl From<PrimalSolveResult> for WarmStartResult {
    fn from(result: PrimalSolveResult) -> Self {
        match result {
            PrimalSolveResult::Optimal(solution) => WarmStartResult::Optimal(solution),
            PrimalSolveResult::IterationLimit(solution) => {
                WarmStartResult::IterationLimit(solution)
            }
            PrimalSolveResult::Unbounded {
                entering,
                direction,
                iterations,
            } => WarmStartResult::Unbounded {
                entering,
                direction,
                iterations,
            },
        }
    }
}

impl From<DualSolveResult> for WarmStartResult {
    fn from(result: DualSolveResult) -> Self {
        match result {
            DualSolveResult::Optimal(solution) => WarmStartResult::Optimal(solution),
            DualSolveResult::IterationLimit(solution) => WarmStartResult::IterationLimit(solution),
            DualSolveResult::Infeasible {
                leaving,
                pivot_row,
                certificate,
                iterations,
            } => WarmStartResult::Infeasible {
                leaving,
                pivot_row,
                certificate,
                iterations,
            },
        }
    }
}

impl WarmStart {
    /// Continue optimization from the selected warm-start state.
    ///
    /// This method does not introduce a separate warm-start algorithm. It calls
    /// primal revised simplex when the reused basis is primal feasible and dual
    /// revised simplex when the reused basis is only dual feasible.
    pub fn solve(
        &mut self,
        trace: &mut impl SimplexTrace,
    ) -> Result<WarmStartResult, SimplexError> {
        match self {
            WarmStart::Primal(simplex) => simplex.solve(trace).map(WarmStartResult::from),
            WarmStart::Dual(simplex) => simplex
                .solve(trace)
                .map(WarmStartResult::from)
                .map_err(SimplexError::from),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum WarmStartError {
    Problem(StandardFormError),
    NoReusableBasis {
        primal: PrimalSimplexError,
        dual: DualSimplexError,
    },
}

#[katexit::katexit]
/// Rebuild a simplex state from an existing basis for a modified LP.
///
/// Given a modified standard-form LP and a basis index set $I$, this function
/// first rebuilds $B = A_I$ from the new matrix data. If the resulting basic
/// solution
///
/// $$
/// x_I = B^{-1}b
/// $$
///
/// is primal feasible, it returns [`WarmStart::Primal`]. Otherwise it checks
/// whether the same basis is dual feasible,
///
/// $$
/// r_j = c_j - A_j^T y \ge -\epsilon \quad (j \notin I),
/// $$
///
/// and returns [`WarmStart::Dual`] when dual simplex can repair the changed
/// right-hand side or constraints. If neither invariant holds, the basis cannot
/// be reused directly and a fresh Phase I construction is needed.
///
/// This function refactorizes the basis from the modified LP. It does not yet
/// preserve an existing LU factorization across LP edits.
pub fn warm_start(
    lp: StandardFormLp,
    basis_indices: Vec<usize>,
    options: RevisedSimplexOptions,
) -> Result<WarmStart, WarmStartError> {
    let state =
        RevisedSimplexState::new(lp, basis_indices, options).map_err(WarmStartError::Problem)?;

    if let Some((position, value)) =
        primal_infeasible_basic_value(state.lp(), state.basis(), state.options().pivot_tolerance)
            .map_err(WarmStartError::Problem)?
    {
        let primal = PrimalSimplexError::PrimalInfeasibleInitialBasis { position, value };
        match DualRevisedSimplex::from_state(state) {
            Ok(simplex) => Ok(WarmStart::Dual(simplex)),
            Err(dual) => Err(WarmStartError::NoReusableBasis { primal, dual }),
        }
    } else {
        Ok(WarmStart::Primal(
            RevisedSimplex::from_state(state).expect("state was already checked primal feasible"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::simplex::NoTrace;

    #[test]
    fn warm_start_uses_primal_when_reused_basis_is_primal_feasible() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![1.0, 2.0],
            array![10.0, -5.0],
        )
        .unwrap();

        let start = warm_start(lp, vec![0, 1], RevisedSimplexOptions::default()).unwrap();

        match start {
            WarmStart::Primal(simplex) => assert_eq!(simplex.basis().indices(), &[0, 1]),
            WarmStart::Dual(_) => panic!("expected primal warm start"),
        }
    }

    #[test]
    fn warm_start_uses_dual_when_reused_basis_is_only_dual_feasible() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![1.0, -2.0],
            array![0.0, 0.0],
        )
        .unwrap();

        let start = warm_start(lp, vec![0, 1], RevisedSimplexOptions::default()).unwrap();

        match start {
            WarmStart::Primal(_) => panic!("expected dual warm start"),
            WarmStart::Dual(simplex) => assert_eq!(simplex.basis().indices(), &[0, 1]),
        }
    }

    #[test]
    fn warm_start_reports_basis_that_cannot_be_reused() {
        let lp = StandardFormLp::new(array![[1.0, 0.0]], array![-1.0], array![0.0, -1.0]).unwrap();

        let error = warm_start(lp, vec![0], RevisedSimplexOptions::default()).unwrap_err();

        assert!(matches!(error, WarmStartError::NoReusableBasis { .. }));
    }

    #[test]
    fn warm_start_solves_primal_selected_state() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![1.0, 2.0],
            array![10.0, -5.0],
        )
        .unwrap();
        let mut start = warm_start(lp, vec![0, 1], RevisedSimplexOptions::default()).unwrap();
        let mut trace = NoTrace;

        let result = start.solve(&mut trace).unwrap();

        let WarmStartResult::Optimal(solution) = result else {
            panic!("expected optimal warm-start result");
        };
        assert_eq!(solution.basis_indices, vec![0, 1]);
        assert_eq!(solution.primal, array![1.0, 2.0]);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn warm_start_solves_dual_selected_state() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![1.0, -2.0],
            array![0.0, 0.0],
        )
        .unwrap();
        let mut start = warm_start(lp, vec![0, 1], RevisedSimplexOptions::default()).unwrap();
        let mut trace = NoTrace;

        let result = start.solve(&mut trace).unwrap();

        let WarmStartResult::Infeasible {
            leaving,
            certificate,
            iterations,
            ..
        } = result
        else {
            panic!("expected infeasible warm-start result");
        };
        assert_eq!(leaving.position, 1);
        assert_eq!(certificate.support(1.0e-9), vec![1]);
        assert_eq!(iterations, 0);
    }
}
