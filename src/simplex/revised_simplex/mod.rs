use ndarray::Array1;

use super::{FarkasCertificate, PricedColumn, StandardFormError};
use crate::simplex::dual::SolveResult as DualSolveResult;
use crate::simplex::primal::{
    PhaseOneError, PhaseOneInfeasible, PhaseOneIterationLimit, PrimalSimplexError, SolveResult,
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
///
/// The `dual` vector is the basis dual vector
///
/// $$
/// B^T y = c_I.
/// $$
///
/// For an optimal result, this is an optimal dual solution. For an
/// iteration-limit result, it is still the dual vector associated with the
/// returned basis, but optimality has not been proved.
pub struct SimplexSolution {
    pub primal: Array1<f64>,
    pub dual: Array1<f64>,
    pub objective_value: f64,
    pub basis_indices: Vec<usize>,
    pub iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
/// Infeasibility certificate returned by a simplex solve.
///
/// The certificate always proves infeasibility of the primal standard-form
/// system. If infeasibility was found through Phase I, the auxiliary optimum is
/// stored in `phase_one_objective_value`. If infeasibility was found by dual
/// simplex during reoptimization, this field is `None`.
pub struct SimplexInfeasible {
    pub certificate: FarkasCertificate,
    pub iterations: usize,
    pub phase_one_objective_value: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
/// Outcome of solving a standard-form LP from an automatically constructed
/// initial basis.
pub enum SimplexResult {
    Optimal(SimplexSolution),
    IterationLimit(SimplexSolution),
    PhaseOneIterationLimit(PhaseOneIterationLimit),
    Infeasible(SimplexInfeasible),
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
        iterations: usize,
    },
}

impl From<PhaseOneInfeasible> for SimplexInfeasible {
    fn from(infeasible: PhaseOneInfeasible) -> Self {
        Self {
            certificate: infeasible.certificate,
            iterations: infeasible.iterations,
            phase_one_objective_value: Some(infeasible.objective_value),
        }
    }
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

impl From<DualSolveResult> for SimplexResult {
    fn from(result: DualSolveResult) -> Self {
        match result {
            DualSolveResult::Optimal(solution) => SimplexResult::Optimal(solution),
            DualSolveResult::IterationLimit(solution) => SimplexResult::IterationLimit(solution),
            DualSolveResult::Infeasible {
                certificate,
                iterations,
                ..
            } => SimplexResult::Infeasible(SimplexInfeasible {
                certificate,
                iterations,
                phase_one_objective_value: None,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimplexError {
    Problem(StandardFormError),
    Primal(PrimalSimplexError),
    PhaseOne(PhaseOneError),
}

impl From<StandardFormError> for SimplexError {
    fn from(error: StandardFormError) -> Self {
        SimplexError::Problem(error)
    }
}

impl From<PrimalSimplexError> for SimplexError {
    fn from(error: PrimalSimplexError) -> Self {
        match error {
            PrimalSimplexError::Problem(error) => SimplexError::Problem(error),
            error => SimplexError::Primal(error),
        }
    }
}

impl From<PhaseOneError> for SimplexError {
    fn from(error: PhaseOneError) -> Self {
        SimplexError::PhaseOne(error)
    }
}
