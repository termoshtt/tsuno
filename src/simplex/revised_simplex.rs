use ndarray::Array1;

use super::{Basis, PricedColumn, StandardFormError, StandardFormLp};

#[derive(Clone, Debug)]
pub struct RevisedSimplexOptions {
    pub reduced_cost_tolerance: f64,
}

impl Default for RevisedSimplexOptions {
    fn default() -> Self {
        Self {
            reduced_cost_tolerance: 1.0e-9,
        }
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

    pub fn basic_solution(&self) -> Result<Array1<f64>, StandardFormError> {
        self.lp.basic_solution(&self.basis)
    }

    pub fn dual_variables(&self) -> Result<Array1<f64>, StandardFormError> {
        self.lp.dual_variables(&self.basis)
    }

    pub fn reduced_costs(&self) -> Result<Vec<PricedColumn>, StandardFormError> {
        self.lp.reduced_costs(&self.basis)
    }

    pub fn entering_column(&self) -> Result<Option<PricedColumn>, StandardFormError> {
        self.lp
            .entering_column(&self.basis, self.options.reduced_cost_tolerance)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

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
            },
        )
        .unwrap();

        let entering_column = simplex.entering_column().unwrap();

        assert_eq!(entering_column, None);
    }

    fn improving_slack_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![-1.0, -2.0, 0.0, 0.0],
        )
        .unwrap()
    }

    fn example_lp() -> StandardFormLp {
        StandardFormLp::new(
            array![[2.0, 1.0, 1.0], [1.0, 3.0, 0.0]],
            array![1.0, 1.0],
            array![5.0, 4.0, 1.0],
        )
        .unwrap()
    }
}
