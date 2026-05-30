use ndarray::Array1;

use crate::simplex::{
    Basis, PricedColumn, RevisedSimplexOptions, SimplexSolution, StandardFormError, StandardFormLp,
};

#[derive(Debug)]
#[katexit::katexit]
/// Common basis state shared by primal and dual revised simplex methods.
///
/// For a standard-form LP
///
/// $$
/// \min c^T x
/// \quad \text{s.t.} \quad
/// A x = b,\quad x \ge 0,
/// $$
///
/// and a basis index set $I$, this stores the problem data, the basis
/// representation $B = A_I$, and solver options. This type does not guarantee
/// either the primal invariant $B^{-1}b \ge 0$ or the dual invariant
/// $r_j \ge 0$. The primal and dual simplex wrappers add those invariants.
pub struct RevisedSimplexState {
    lp: StandardFormLp,
    basis: Basis,
    options: RevisedSimplexOptions,
}

impl RevisedSimplexState {
    pub fn new(
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

    pub fn current_solution(
        &self,
        iterations: usize,
    ) -> Result<SimplexSolution, StandardFormError> {
        let basic_solution = self.basic_solution()?;
        let primal = full_primal_solution(self.lp.c().len(), self.basis.indices(), &basic_solution);
        let dual = self.dual_variables()?;
        let objective_value = self.lp.c().dot(&primal);
        Ok(SimplexSolution {
            primal,
            dual,
            objective_value,
            basis_indices: self.basis.indices().to_vec(),
            iterations,
        })
    }

    #[katexit::katexit]
    /// Replace the stored right-hand side while keeping the current basis.
    ///
    /// Changing $b$ in
    ///
    /// $$
    /// A x = b
    /// $$
    ///
    /// changes the basic values $x_I = B^{-1}b$, but it does not change the
    /// basis matrix $B = A_I$ or the reduced costs because $A$ and $c$ are
    /// unchanged. Therefore the LU representation inside [`Basis`] can be
    /// reused directly.
    pub fn replace_rhs(self, rhs: Array1<f64>) -> Result<Self, StandardFormError> {
        let lp = self.lp.replace_rhs(rhs)?;
        Ok(Self { lp, ..self })
    }

    pub(crate) fn solve_basis_column(
        &self,
        column: usize,
    ) -> Result<Array1<f64>, StandardFormError> {
        let column = self.lp.column(column)?.to_owned();
        Ok(self.basis.solve(&column))
    }

    pub(crate) fn solve_transposed_basis_unit(&self, position: usize) -> Array1<f64> {
        let mut unit = Array1::zeros(self.lp.a().nrows());
        unit[position] = 1.0;
        self.basis.solve_transposed(&unit)
    }

    pub(crate) fn replace_basis_column(
        &mut self,
        position: usize,
        column: usize,
    ) -> Result<(), StandardFormError> {
        let entering_column = self.lp.column(column)?.to_owned();
        self.basis
            .replace_column(position, column, &entering_column)
            .map_err(StandardFormError::Basis)
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
