use ndarray::Array1;

use super::{Basis, PricedColumn, StandardFormError, StandardFormLp};

mod phase_one;
mod trace;

pub use phase_one::*;
pub use trace::*;

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
    IterationLimit(SimplexSolution),
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
        iterations: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
/// Outcome of solving a standard-form LP from an automatically constructed
/// initial basis.
pub enum SimplexResult {
    Optimal(SimplexSolution),
    IterationLimit(SimplexSolution),
    PhaseOneIterationLimit(SimplexSolution),
    Infeasible(PhaseOneInfeasible),
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
        iterations: usize,
    },
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
    /// reported as unbounded in that direction.
    ///
    /// If neither condition occurs before
    /// [`RevisedSimplexOptions::max_iterations`] step attempts, this returns
    /// [`SimplexSolveResult::IterationLimit`] with the current basic solution.
    /// That result is deliberately distinct from [`SimplexSolveResult::Optimal`]:
    /// the solver has a valid current basis and solution, but has not proved
    /// optimality.
    pub fn solve(
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

        Ok(SimplexSolveResult::IterationLimit(
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
mod tests;
