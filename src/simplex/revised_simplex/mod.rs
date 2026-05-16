use ndarray::Array1;

use super::{PricedColumn, StandardFormError, StandardFormLp};
use crate::simplex::primal::{
    PhaseOneAuxiliaryProblem, PhaseOneError, PhaseOneInfeasible, PhaseOneIterationLimit,
    PhaseOneResult, RevisedSimplex, SolveResult,
};

mod trace;

pub use trace::*;

#[derive(Clone, Debug)]
pub struct RevisedSimplexOptions {
    pub reduced_cost_tolerance: f64,
    pub pivot_tolerance: f64,
    /// Maximum number of `step` calls attempted by a revised simplex solve.
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
#[katexit::katexit]
/// Solution returned by a simplex solve.
///
/// The `primal` vector is the full decision vector `x`, including both basis
/// and nonbasis components. Nonbasis components are zero in a returned basic
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
/// Outcome of solving a standard-form LP from an automatically constructed
/// initial basis.
pub enum SimplexResult {
    Optimal(SimplexSolution),
    IterationLimit(SimplexSolution),
    PhaseOneIterationLimit(PhaseOneIterationLimit),
    Infeasible(PhaseOneInfeasible),
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
        iterations: usize,
    },
}

impl From<SolveResult> for SimplexResult {
    fn from(result: SolveResult) -> Self {
        match result {
            SolveResult::Optimal(solution) => SimplexResult::Optimal(solution),
            SolveResult::IterationLimit(solution) => SimplexResult::IterationLimit(solution),
            SolveResult::Unbounded {
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
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimplexError {
    Problem(StandardFormError),
    PhaseOne(PhaseOneError),
}

impl From<StandardFormError> for SimplexError {
    fn from(error: StandardFormError) -> Self {
        SimplexError::Problem(error)
    }
}

impl From<PhaseOneError> for SimplexError {
    fn from(error: PhaseOneError) -> Self {
        SimplexError::PhaseOne(error)
    }
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
) -> Result<SimplexResult, SimplexError> {
    match PhaseOneAuxiliaryProblem::new(&lp).solve(options.clone(), trace)? {
        PhaseOneResult::Feasible { basis_indices } => {
            let mut simplex = RevisedSimplex::with_options(lp, basis_indices, options)?;
            trace.phase_started(SimplexTracePhase::PhaseTwo);
            simplex.solve(trace).map(SimplexResult::from)
        }
        PhaseOneResult::Infeasible(infeasible) => Ok(SimplexResult::Infeasible(infeasible)),
        PhaseOneResult::IterationLimit(limit) => Ok(SimplexResult::PhaseOneIterationLimit(limit)),
    }
}
