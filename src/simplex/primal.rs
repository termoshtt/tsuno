use ndarray::Array1;

use super::revised_simplex::{
    RevisedSimplexOptions, SimplexError, SimplexSolution, SimplexTrace, SimplexTraceEvent,
};
use super::{Basis, PricedColumn, StandardFormError, StandardFormLp};

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
/// A primal revised simplex step keeps the basic solution primal feasible,
///
/// $$
/// x_I = B^{-1} b \ge 0,
/// $$
///
/// and repairs a negative reduced cost. It first chooses a nonbasis column
/// with the most negative reduced cost, computes
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
    pub(crate) lp: StandardFormLp,
    pub(crate) basis: Basis,
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

    /// Select the nonbasis column with the most negative reduced cost.
    ///
    /// This is a state-level wrapper around
    /// [`StandardFormLp::most_negative_reduced_cost`] using
    /// [`RevisedSimplexOptions::reduced_cost_tolerance`].
    pub fn most_negative_reduced_cost(&self) -> Result<Option<PricedColumn>, StandardFormError> {
        self.lp
            .most_negative_reduced_cost(&self.basis, self.options.reduced_cost_tolerance)
    }

    pub fn step(&mut self) -> Result<Step, StandardFormError> {
        let Some(entering) = self.most_negative_reduced_cost()? else {
            return Ok(Step::Optimal);
        };

        let basic_solution = self.basic_solution()?;
        let direction = self.pivot_direction(entering.column)?;
        let Some(leaving_position) =
            primal_minimum_ratio_test(&basic_solution, &direction, self.options.pivot_tolerance)
        else {
            return Ok(Step::Unbounded {
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

        Ok(Step::Pivoted {
            entering,
            leaving,
            direction,
        })
    }

    fn pivot_direction(&self, entering_column: usize) -> Result<Array1<f64>, StandardFormError> {
        let column = self.lp.column(entering_column)?.to_owned();
        Ok(self.basis.solve(&column))
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
        for iteration in 0..self.options.max_iterations {
            trace.step_started(iteration, self.basis.indices());
            let step = self.step()?;
            trace.step_completed(SimplexTraceEvent {
                iteration,
                step: &step,
                basis_after: self.basis.indices(),
            });

            match step {
                Step::Optimal => {
                    return Ok(SolveResult::Optimal(self.current_solution(iteration)?));
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
