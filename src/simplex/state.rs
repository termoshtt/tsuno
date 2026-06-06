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

    #[katexit::katexit]
    /// Replace the stored cost vector while keeping the current basis.
    ///
    /// Changing $c$ in
    ///
    /// $$
    /// \min c^T x
    /// $$
    ///
    /// changes the basis costs $c_I$, the dual variables $B^{-T}c_I$, and the
    /// reduced costs. It does not change $A$, $b$, or the basis matrix
    /// $B=A_I$, so the basic values $x_I = B^{-1}b$ and the LU representation
    /// inside [`Basis`] can be reused directly.
    pub fn replace_cost(self, cost: Array1<f64>) -> Result<Self, StandardFormError> {
        let lp = self.lp.replace_cost(cost)?;
        Ok(Self { lp, ..self })
    }

    #[katexit::katexit]
    /// Replace a nonbasis column while keeping the current basis.
    ///
    /// If $j \notin I$, changing $A_j$ and $c_j$ does not change the basis
    /// matrix $B=A_I$ or the current basic values $x_I = B^{-1}b$. It may
    /// change the reduced cost of column $j$, so primal revised simplex can
    /// reoptimize from the updated state. Callers must reject basis columns
    /// before using this method.
    pub(crate) fn replace_nonbasis_column(
        self,
        column: usize,
        values: Array1<f64>,
        cost: f64,
    ) -> Result<Self, StandardFormError> {
        let lp = self.lp.replace_column(column, values, cost)?;
        Ok(Self { lp, ..self })
    }

    #[katexit::katexit]
    /// Replace a column and rebuild the current basis representation.
    ///
    /// If $j \in I$, changing $A_j$ changes the basis matrix $B=A_I$. This
    /// method updates the stored LP data and then rebuilds [`Basis`] from the
    /// same basis index set against the updated matrix. It is a full
    /// refactorization path rather than an eta update.
    pub(crate) fn replace_column_and_refactor_basis(
        self,
        column: usize,
        values: Array1<f64>,
        cost: f64,
    ) -> Result<Self, StandardFormError> {
        let Self { lp, basis, options } = self;
        let basis_indices = basis.indices().to_vec();
        let lp = lp.replace_column(column, values, cost)?;
        let basis = lp.basis(basis_indices)?;
        Ok(Self { lp, basis, options })
    }

    #[katexit::katexit]
    /// Add a nonbasis column while keeping the current basis.
    ///
    /// Appending a new column $A_j$ with cost $c_j$ does not change the current
    /// basis matrix $B=A_I$, because the new column is not in $I$. It may have
    /// negative reduced cost, so primal revised simplex can reoptimize from the
    /// updated state. The returned `usize` is the new column index $j$.
    pub(crate) fn add_nonbasis_column(
        self,
        values: Array1<f64>,
        cost: f64,
    ) -> Result<(Self, usize), StandardFormError> {
        let (lp, column) = self.lp.add_column(values, cost)?;
        Ok((Self { lp, ..self }, column))
    }

    #[katexit::katexit]
    /// Remove a nonbasis column and remap the current basis indices.
    ///
    /// If $j \notin I$, removing $A_j$ does not change the basis matrix
    /// $B=A_I$. The LP removes the column by moving the old last column into
    /// position $j$, so only the old last column's basis index may need to be
    /// remapped.
    pub(crate) fn remove_nonbasis_column(self, column: usize) -> Result<Self, StandardFormError> {
        let Self { lp, basis, options } = self;
        let last_column = lp.a().ncols() - 1;
        let lp = lp.remove_column(column)?;
        let basis = basis
            .remap_indices_after_swap_remove_column(column, last_column)
            .map_err(StandardFormError::Basis)?;
        Ok(Self { lp, basis, options })
    }

    #[katexit::katexit]
    /// Remove a basis column and try to repair the basis by refactorization.
    ///
    /// If $j \in I$, removing $A_j$ removes one column from $B=A_I$. This
    /// method removes the column from the LP, remaps any remaining basis column
    /// moved by the swap-remove operation, and searches for a replacement
    /// column for the removed basis position. Each candidate basis is rebuilt
    /// from the updated matrix; the first full-rank candidate is returned.
    pub(crate) fn remove_basis_column_and_refactor(
        self,
        column: usize,
    ) -> Result<Option<Self>, StandardFormError> {
        let Self { lp, basis, options } = self;
        let Some(position) = basis.indices().iter().position(|&index| index == column) else {
            let state = Self { lp, basis, options };
            return state.remove_nonbasis_column(column).map(Some);
        };
        let last_column = lp.a().ncols() - 1;
        let partial_basis = remap_basis_indices_after_swap_remove_without_removed(
            basis.indices(),
            column,
            last_column,
        );
        let lp = lp.remove_column(column)?;

        for replacement in 0..lp.a().ncols() {
            if partial_basis.contains(&replacement) {
                continue;
            }
            let mut basis_indices = partial_basis.clone();
            basis_indices.insert(position, replacement);
            let basis = lp.basis(basis_indices)?;
            if basis.lu().row_permutation().len() == lp.a().nrows() {
                return Ok(Some(Self { lp, basis, options }));
            }
        }

        Ok(None)
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

fn remap_basis_indices_after_swap_remove_without_removed(
    indices: &[usize],
    column: usize,
    last_column: usize,
) -> Vec<usize> {
    indices
        .iter()
        .filter_map(|&index| {
            if index == column {
                None
            } else if index == last_column {
                Some(column)
            } else {
                Some(index)
            }
        })
        .collect()
}
