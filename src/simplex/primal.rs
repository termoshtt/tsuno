use ndarray::Array1;

use super::revised_simplex::{
    RevisedSimplexOptions, SimplexError, SimplexResult, SimplexSolution, SimplexTrace,
    SimplexTraceEvent, SimplexTracePhase, SimplexTraceStep,
};
use super::{Basis, PricedColumn, RevisedSimplexState, StandardFormError, StandardFormLp};

mod phase_one;

pub(crate) use phase_one::PhaseOneResult;
pub use phase_one::{
    PhaseOneAuxiliaryProblem, PhaseOneError, PhaseOneInfeasible, PhaseOneIterationLimit,
};

#[derive(Clone, Debug, PartialEq)]
/// Original column that leaves the basis in a pivoted primal simplex step.
///
/// The `column` field is the original column index in `A`, not the internal
/// position inside the current ordered basis. The `step_length` field is the
/// primal ratio-test value at that leaving column.
pub struct LeavingColumn {
    pub column: usize,
    pub step_length: f64,
}

/// Internal leaving basis position selected by the primal minimum ratio test.
#[derive(Clone, Debug, PartialEq)]
struct LeavingPosition {
    position: usize,
    step_length: f64,
}

#[derive(Clone, Debug, PartialEq)]
/// Error returned while constructing a primal revised simplex state.
pub enum PrimalSimplexError {
    Problem(StandardFormError),
    PrimalInfeasibleInitialBasis { position: usize, value: f64 },
}

impl From<StandardFormError> for PrimalSimplexError {
    fn from(error: StandardFormError) -> Self {
        PrimalSimplexError::Problem(error)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
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
/// Outcome of repeatedly applying primal revised simplex steps.
pub enum SolveResult {
    Optimal(SimplexSolution),
    IterationLimit(SimplexSolution),
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
        iterations: usize,
    },
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
/// This module exposes one full-featured primal solve entry point. Pass
/// [`RevisedSimplexOptions::default`] for default tolerances and iteration
/// limits, and pass [`crate::simplex::NoTrace`] when no trace collection is
/// needed.
pub fn solve(
    lp: StandardFormLp,
    options: RevisedSimplexOptions,
    trace: &mut impl SimplexTrace,
) -> Result<SimplexResult, SimplexError> {
    match PhaseOneAuxiliaryProblem::new(lp.clone()).solve(options.clone(), trace)? {
        PhaseOneResult::Feasible { basis_indices } => {
            let mut simplex = RevisedSimplex::new(lp, basis_indices, options)?;
            trace.phase_started(SimplexTracePhase::PhaseTwo);
            simplex.solve(trace).map(SimplexResult::from)
        }
        PhaseOneResult::Infeasible(infeasible) => Ok(SimplexResult::Infeasible(infeasible)),
        PhaseOneResult::IterationLimit(limit) => Ok(SimplexResult::PhaseOneIterationLimit(limit)),
    }
}

#[katexit::katexit]
/// State for the primal revised simplex method.
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
/// # Invariant
///
/// A value of this type has a primal-feasible basis:
///
/// $$
/// x_I = B^{-1} b \ge -\epsilon.
/// $$
///
/// The constructors reject a basis that violates this condition, using
/// [`RevisedSimplexOptions::pivot_tolerance`] as $\epsilon$.
///
/// # Step
///
/// A primal revised simplex step keeps this invariant and repairs a negative
/// reduced cost. It first chooses a nonbasis column with the most negative
/// reduced cost, computes
///
/// $$
/// d = B^{-1} A_q,
/// $$
///
/// and then applies the primal minimum ratio test
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
    state: RevisedSimplexState,
}

impl RevisedSimplex {
    pub fn new(
        lp: StandardFormLp,
        basis_indices: Vec<usize>,
        options: RevisedSimplexOptions,
    ) -> Result<Self, PrimalSimplexError> {
        let state = RevisedSimplexState::new(lp, basis_indices, options)?;
        Self::from_state(state)
    }

    pub fn from_state(state: RevisedSimplexState) -> Result<Self, PrimalSimplexError> {
        if let Some((position, value)) = primal_infeasible_basic_value(
            state.lp(),
            state.basis(),
            state.options().pivot_tolerance,
        )? {
            return Err(PrimalSimplexError::PrimalInfeasibleInitialBasis { position, value });
        }
        Ok(Self { state })
    }

    pub fn lp(&self) -> &StandardFormLp {
        self.state.lp()
    }

    pub fn basis(&self) -> &Basis {
        self.state.basis()
    }

    pub fn options(&self) -> &RevisedSimplexOptions {
        self.state.options()
    }

    pub fn into_state(self) -> RevisedSimplexState {
        self.state
    }

    /// Compute the current basic solution values.
    ///
    /// This is a state-level wrapper around [`StandardFormLp::basic_solution`].
    /// See that method for the mathematical definition of the returned
    /// $x_I = B^{-1} b$ vector.
    pub fn basic_solution(&self) -> Result<Array1<f64>, StandardFormError> {
        self.state.basic_solution()
    }

    /// Compute the dual variables for the current basis.
    ///
    /// This is a state-level wrapper around [`StandardFormLp::dual_variables`].
    /// See that method for the transposed basis system $B^T y = c_I$.
    pub fn dual_variables(&self) -> Result<Array1<f64>, StandardFormError> {
        self.state.dual_variables()
    }

    /// Compute reduced costs for all current nonbasis columns.
    ///
    /// This is a state-level wrapper around [`StandardFormLp::reduced_costs`].
    /// See that method for the definition of $r_j = c_j - A_j^T y$.
    pub fn reduced_costs(&self) -> Result<Vec<PricedColumn>, StandardFormError> {
        self.state.reduced_costs()
    }

    /// Select the nonbasis column with the most negative reduced cost.
    ///
    /// This is a state-level wrapper around
    /// [`StandardFormLp::most_negative_reduced_cost`] using
    /// [`RevisedSimplexOptions::reduced_cost_tolerance`].
    pub fn most_negative_reduced_cost(&self) -> Result<Option<PricedColumn>, StandardFormError> {
        self.lp()
            .most_negative_reduced_cost(self.basis(), self.options().reduced_cost_tolerance)
    }

    pub fn step(&mut self) -> Result<Step, StandardFormError> {
        let Some(entering) = self.most_negative_reduced_cost()? else {
            return Ok(Step::Optimal);
        };

        let basic_solution = self.basic_solution()?;
        let direction = self.pivot_direction(entering.column)?;
        let Some(leaving_position) =
            primal_minimum_ratio_test(&basic_solution, &direction, self.options().pivot_tolerance)
        else {
            return Ok(Step::Unbounded {
                entering,
                direction,
            });
        };

        let leaving = LeavingColumn {
            column: self.basis().indices()[leaving_position.position],
            step_length: leaving_position.step_length,
        };

        self.state
            .replace_basis_column(leaving_position.position, entering.column)?;

        Ok(Step::Pivoted {
            entering,
            leaving,
            direction,
        })
    }

    fn pivot_direction(&self, entering_column: usize) -> Result<Array1<f64>, StandardFormError> {
        self.state.solve_basis_column(entering_column)
    }

    pub(crate) fn basis_direction(&self, column: usize) -> Result<Array1<f64>, StandardFormError> {
        self.state.solve_basis_column(column)
    }

    pub(crate) fn replace_basis_column(
        &mut self,
        position: usize,
        column: usize,
    ) -> Result<(), StandardFormError> {
        self.state.replace_basis_column(position, column)
    }

    /// Repeatedly apply primal revised simplex steps until termination.
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
    /// reported as unbounded in that direction.
    ///
    /// If neither condition occurs before
    /// [`RevisedSimplexOptions::max_iterations`] step attempts, this returns
    /// [`SolveResult::IterationLimit`] with the current basic solution.
    /// That result is deliberately distinct from [`SolveResult::Optimal`]:
    /// the solver has a valid current basis and solution, but has not proved
    /// optimality.
    pub fn solve(&mut self, trace: &mut impl SimplexTrace) -> Result<SolveResult, SimplexError> {
        for iteration in 0..self.options().max_iterations {
            trace.step_started(iteration, self.basis().indices());
            let step = self.step()?;
            trace.step_completed(SimplexTraceEvent {
                iteration,
                step: SimplexTraceStep::Primal(&step),
                basis_after: self.basis().indices(),
            });

            match step {
                Step::Optimal => {
                    return Ok(SolveResult::Optimal(
                        self.state.current_solution(iteration)?,
                    ));
                }
                Step::Unbounded {
                    entering,
                    direction,
                } => {
                    return Ok(SolveResult::Unbounded {
                        entering,
                        direction,
                        iterations: iteration,
                    });
                }
                Step::Pivoted { .. } => {}
            }
        }

        Ok(SolveResult::IterationLimit(
            self.state.current_solution(self.options().max_iterations)?,
        ))
    }
}

pub(crate) fn primal_infeasible_basic_value(
    lp: &StandardFormLp,
    basis: &Basis,
    tolerance: f64,
) -> Result<Option<(usize, f64)>, StandardFormError> {
    let tolerance = tolerance.max(0.0);
    Ok(lp
        .basic_solution(basis)?
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value < -tolerance)
        .min_by(|left, right| left.1.total_cmp(&right.1)))
}

fn primal_minimum_ratio_test(
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
mod tests;
