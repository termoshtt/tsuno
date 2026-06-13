use ndarray::Array1;

use crate::simplex::dual::{DualRevisedSimplex, DualSimplexError};
use crate::simplex::primal::{PrimalSimplexError, RevisedSimplex, primal_infeasible_basic_value};
use crate::simplex::{
    RevisedSimplexOptions, RevisedSimplexState, SimplexError, SimplexResult, SimplexTrace,
    StandardFormError, StandardFormLp,
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

#[derive(Debug)]
/// A terminal simplex result together with the state needed for reoptimization.
///
/// This type is the reusable counterpart of a solve result. It only exposes a
/// state after simplex has reached a terminal condition for that state, so
/// callers do not have to handle an arbitrary intermediate
/// [`RevisedSimplexState`] whose primal or dual invariant is unknown.
pub struct SolvedSimplex {
    state: Box<RevisedSimplexState>,
    result: SimplexResult,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReoptimizationError {
    WarmStart(WarmStartError),
    Primal(PrimalSimplexError),
    Dual(DualSimplexError),
    NoReplacementBasisAfterColumnRemoval { column: usize },
    Simplex(SimplexError),
}

impl From<WarmStartError> for ReoptimizationError {
    fn from(error: WarmStartError) -> Self {
        ReoptimizationError::WarmStart(error)
    }
}

impl From<SimplexError> for ReoptimizationError {
    fn from(error: SimplexError) -> Self {
        ReoptimizationError::Simplex(error)
    }
}

impl From<PrimalSimplexError> for ReoptimizationError {
    fn from(error: PrimalSimplexError) -> Self {
        ReoptimizationError::Primal(error)
    }
}

impl From<DualSimplexError> for ReoptimizationError {
    fn from(error: DualSimplexError) -> Self {
        ReoptimizationError::Dual(error)
    }
}

impl From<StandardFormError> for ReoptimizationError {
    fn from(error: StandardFormError) -> Self {
        ReoptimizationError::Simplex(SimplexError::from(error))
    }
}

impl SolvedSimplex {
    pub(crate) fn new(state: RevisedSimplexState, result: SimplexResult) -> Self {
        Self {
            state: Box::new(state),
            result,
        }
    }

    /// Return the terminal result associated with the reusable state.
    pub fn result(&self) -> &SimplexResult {
        &self.result
    }

    /// Return the reusable revised-simplex state.
    pub fn state(&self) -> &RevisedSimplexState {
        self.state.as_ref()
    }

    pub fn into_state(self) -> RevisedSimplexState {
        *self.state
    }

    #[katexit::katexit]
    /// Replace the right-hand side and reoptimize immediately.
    ///
    /// If the previous state was optimal, changing only $b$ preserves the dual
    /// variables and reduced costs because $A$, $c$, and $B=A_I$ are unchanged.
    /// The new basic values
    ///
    /// $$
    /// x_I = B^{-1} b
    /// $$
    ///
    /// may become negative, so this method runs dual revised simplex from the
    /// updated state before returning another [`SolvedSimplex`].
    pub fn reoptimize_with_rhs(
        self,
        rhs: Array1<f64>,
        trace: &mut impl SimplexTrace,
    ) -> Result<Self, ReoptimizationError> {
        let state = self.into_state().replace_rhs(rhs)?;
        let mut simplex = DualRevisedSimplex::from_state(state)?;
        let result = simplex.solve(trace)?;
        let state = simplex.into_state();
        Ok(SolvedSimplex::new(state, SimplexResult::from(result)))
    }

    #[katexit::katexit]
    /// Replace the objective cost vector and reoptimize immediately.
    ///
    /// Changing only $c$ preserves the current basic values because $A$, $b$,
    /// and $B=A_I$ are unchanged:
    ///
    /// $$
    /// x_I = B^{-1}b.
    /// $$
    ///
    /// Therefore a previously primal-feasible terminal state remains
    /// primal-feasible. The dual variables and reduced costs can change, so
    /// this method runs primal revised simplex from the updated state before
    /// returning another [`SolvedSimplex`].
    pub fn reoptimize_with_cost(
        self,
        cost: Array1<f64>,
        trace: &mut impl SimplexTrace,
    ) -> Result<Self, ReoptimizationError> {
        let state = self.into_state().replace_cost(cost)?;
        let mut simplex = RevisedSimplex::from_state(state)?;
        let result = simplex.solve(trace)?;
        let state = simplex.into_state();
        Ok(SolvedSimplex::new(state, SimplexResult::from(result)))
    }

    #[katexit::katexit]
    /// Replace one original column and reoptimize immediately.
    ///
    /// The caller only specifies the original column index $j$, the new
    /// constraint column $A_j$, and the new cost $c_j$. This method checks
    /// whether $j$ is currently in the basis.
    ///
    /// If $j \notin I$, the basis matrix $B=A_I$ is unchanged, so the current
    /// basic values remain primal-feasible:
    ///
    /// $$
    /// x_I = B^{-1}b.
    /// $$
    ///
    /// The changed column may have a new reduced cost, so this method runs
    /// primal revised simplex from the updated state.
    ///
    /// If $j \in I$, updating the column changes $B$ itself. In that case this
    /// method rebuilds the basis representation from the updated matrix and
    /// then chooses primal or dual revised simplex according to the rebuilt
    /// state's invariant.
    pub fn reoptimize_with_column(
        self,
        column: usize,
        values: Array1<f64>,
        cost: f64,
        trace: &mut impl SimplexTrace,
    ) -> Result<Self, ReoptimizationError> {
        let state = self.into_state();
        if state.basis().indices().contains(&column) {
            let state = state.replace_column_and_refactor_basis(column, values, cost)?;
            return WarmStart::from_state(state)?.solve_reusable(trace);
        }

        let state = state.replace_nonbasis_column(column, values, cost)?;
        let mut simplex = RevisedSimplex::from_state(state)?;
        let result = simplex.solve(trace)?;
        let state = simplex.into_state();
        Ok(SolvedSimplex::new(state, SimplexResult::from(result)))
    }

    #[katexit::katexit]
    /// Add a new original column and reoptimize immediately.
    ///
    /// Appending a column creates a new nonbasis variable. The current basis
    /// matrix $B=A_I$ and basic values are unchanged:
    ///
    /// $$
    /// x_I = B^{-1}b.
    /// $$
    ///
    /// The new column may have negative reduced cost, so this method runs
    /// primal revised simplex from the updated state. The returned `usize` is
    /// the new column index.
    pub fn reoptimize_with_added_column(
        self,
        values: Array1<f64>,
        cost: f64,
        trace: &mut impl SimplexTrace,
    ) -> Result<(Self, usize), ReoptimizationError> {
        let (state, column) = self.into_state().add_nonbasis_column(values, cost)?;
        let mut simplex = RevisedSimplex::from_state(state)?;
        let result = simplex.solve(trace)?;
        let state = simplex.into_state();
        Ok((
            SolvedSimplex::new(state, SimplexResult::from(result)),
            column,
        ))
    }

    #[katexit::katexit]
    /// Remove one original column and reoptimize immediately.
    ///
    /// The caller specifies the original column index $j$. If $j \notin I$,
    /// the basis matrix is unchanged. The LP removes the column by moving the
    /// old last column into position $j$, so this method remaps the basis index
    /// only when that old last column was basic. It then runs primal revised
    /// simplex from the updated state.
    ///
    /// If $j \in I$, the basis matrix loses one column. This method removes
    /// the column, searches for a replacement basis column in the updated LP,
    /// refactorizes the repaired basis, and then chooses primal or dual revised
    /// simplex according to the repaired state's invariant.
    pub fn reoptimize_without_column(
        self,
        column: usize,
        trace: &mut impl SimplexTrace,
    ) -> Result<Self, ReoptimizationError> {
        let state = self.into_state();
        if state.basis().indices().contains(&column) {
            let Some(state) = state.remove_basis_column_and_refactor(column)? else {
                return Err(ReoptimizationError::NoReplacementBasisAfterColumnRemoval { column });
            };
            return WarmStart::from_state(state)?.solve_reusable(trace);
        }

        let state = state.remove_nonbasis_column(column)?;
        let mut simplex = RevisedSimplex::from_state(state)?;
        let result = simplex.solve(trace)?;
        let state = simplex.into_state();
        Ok(SolvedSimplex::new(state, SimplexResult::from(result)))
    }

    #[katexit::katexit]
    /// Add one less-than-or-equal constraint with a slack basis variable and
    /// reoptimize immediately.
    ///
    /// This is the fast reoptimization path for adding a constraint
    ///
    /// $$
    /// a^T x \le \beta.
    /// $$
    ///
    /// The updated standard-form LP is
    ///
    /// $$
    /// \begin{bmatrix}
    /// A & 0 \\
    /// a^T & 1
    /// \end{bmatrix}
    /// \begin{bmatrix}
    /// x \\
    /// s
    /// \end{bmatrix}
    /// =
    /// \begin{bmatrix}
    /// b \\
    /// \beta
    /// \end{bmatrix},
    /// \qquad x,s \ge 0.
    /// $$
    ///
    /// The new slack column is added to the basis. If the previous basis was
    /// $I$, the new basis matrix is
    ///
    /// $$
    /// B^+ =
    /// \begin{bmatrix}
    /// B & 0 \\
    /// a_I^T & 1
    /// \end{bmatrix}.
    /// $$
    ///
    /// This block representation reuses the previous basis solve. Its basic
    /// slack value is
    ///
    /// $$
    /// s = \beta - a^T x.
    /// $$
    ///
    /// If $s \ge 0$, the old optimum remains feasible and dual simplex returns
    /// optimality immediately. If $s < 0$, the slack is the only newly
    /// introduced primal infeasibility while reduced costs stay dual feasible,
    /// so dual revised simplex can reoptimize from this state.
    pub fn reoptimize_with_less_equal_constraint(
        self,
        coefficients: Array1<f64>,
        upper_bound: f64,
        trace: &mut impl SimplexTrace,
    ) -> Result<(Self, usize), ReoptimizationError> {
        let (state, slack_column) = self
            .into_state()
            .add_less_equal_constraint_with_slack_basis(coefficients, upper_bound)?;
        let mut simplex = DualRevisedSimplex::from_state(state)?;
        let result = simplex.solve(trace)?;
        let state = simplex.into_state();
        Ok((
            SolvedSimplex::new(state, SimplexResult::from(result)),
            slack_column,
        ))
    }
}

impl WarmStart {
    /// Continue optimization from the selected warm-start state.
    ///
    /// This method does not introduce a separate warm-start algorithm. It calls
    /// primal revised simplex when the reused basis is primal feasible and dual
    /// revised simplex when the reused basis is only dual feasible.
    pub fn solve(&mut self, trace: &mut impl SimplexTrace) -> Result<SimplexResult, SimplexError> {
        match self {
            WarmStart::Primal(simplex) => simplex.solve(trace).map(SimplexResult::from),
            WarmStart::Dual(simplex) => simplex
                .solve(trace)
                .map(SimplexResult::from)
                .map_err(SimplexError::from),
        }
    }

    /// Continue optimization and return the terminal result with reusable state.
    pub fn solve_reusable(
        self,
        trace: &mut impl SimplexTrace,
    ) -> Result<SolvedSimplex, ReoptimizationError> {
        match self {
            WarmStart::Primal(mut simplex) => {
                let result = simplex.solve(trace)?;
                let state = simplex.into_state();
                Ok(SolvedSimplex::new(state, SimplexResult::from(result)))
            }
            WarmStart::Dual(mut simplex) => {
                let result = simplex.solve(trace)?;
                let state = simplex.into_state();
                Ok(SolvedSimplex::new(state, SimplexResult::from(result)))
            }
        }
    }

    pub fn from_state(state: RevisedSimplexState) -> Result<Self, WarmStartError> {
        if let Some((position, value)) = primal_infeasible_basic_value(
            state.lp(),
            state.basis(),
            state.options().pivot_tolerance,
        )
        .map_err(WarmStartError::Problem)?
        {
            let primal = PrimalSimplexError::PrimalInfeasibleInitialBasis { position, value };
            match DualRevisedSimplex::from_state(state) {
                Ok(simplex) => Ok(WarmStart::Dual(simplex)),
                Err(dual) => Err(WarmStartError::NoReusableBasis { primal, dual }),
            }
        } else {
            match RevisedSimplex::from_state(state) {
                Ok(simplex) => Ok(WarmStart::Primal(simplex)),
                Err(PrimalSimplexError::Problem(error)) => Err(WarmStartError::Problem(error)),
                Err(error) => Err(WarmStartError::UnexpectedPrimalError(error)),
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum WarmStartError {
    Problem(StandardFormError),
    UnexpectedPrimalError(PrimalSimplexError),
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
    WarmStart::from_state(state)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::simplex::{NoTrace, primal};

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

        let SimplexResult::Optimal(solution) = result else {
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

        let SimplexResult::Infeasible(certificate) = result else {
            panic!("expected infeasible warm-start result");
        };
        assert_eq!(certificate.support(1.0e-9), vec![1]);
    }

    #[test]
    fn solved_simplex_reoptimizes_after_rhs_replacement() {
        let lp = StandardFormLp::new(
            array![[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]],
            array![1.0, 1.0],
            array![0.0, 1.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };
        assert!(matches!(solved.result(), SimplexResult::Optimal(_)));

        let resolved = solved
            .reoptimize_with_rhs(array![-1.0, 1.0], &mut trace)
            .unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(solution.basis_indices, vec![1, 2]);
        assert_eq!(solution.primal, array![0.0, 1.0, 0.0]);
        assert_eq!(resolved.state().lp().b(), &array![-1.0, 1.0]);
    }

    #[test]
    fn solved_simplex_reoptimizes_after_cost_replacement() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };
        let SimplexResult::Optimal(solution) = solved.result() else {
            panic!("expected initial optimal result");
        };
        assert_eq!(solution.basis_indices, vec![2, 3]);
        assert_eq!(solution.primal, array![0.0, 0.0, 4.0, 3.0]);

        let resolved = solved
            .reoptimize_with_cost(array![-1.0, -2.0, 0.0, 0.0], &mut trace)
            .unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(solution.basis_indices, vec![0, 1]);
        assert_eq!(solution.primal, array![4.0, 3.0, 0.0, 0.0]);
        assert_eq!(resolved.state().lp().c(), &array![-1.0, -2.0, 0.0, 0.0]);
    }

    #[test]
    fn solved_simplex_reoptimizes_after_nonbasis_column_replacement() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };
        let SimplexResult::Optimal(solution) = solved.result() else {
            panic!("expected initial optimal result");
        };
        assert_eq!(solution.basis_indices, vec![2, 3]);

        let resolved = solved
            .reoptimize_with_column(0, array![1.0, 1.0], -1.0, &mut trace)
            .unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(solution.basis_indices, vec![2, 0]);
        assert_eq!(solution.primal, array![3.0, 0.0, 1.0, 0.0]);
        assert_eq!(resolved.state().lp().a().column(0), array![1.0, 1.0]);
        assert_eq!(resolved.state().lp().c()[0], -1.0);
    }

    #[test]
    fn solved_simplex_reoptimizes_after_basis_column_replacement() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };

        let resolved = solved
            .reoptimize_with_column(2, array![-1.0, 0.0], 0.0, &mut trace)
            .unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(solution.basis_indices, vec![0, 3]);
        assert_eq!(solution.primal, array![4.0, 0.0, 0.0, 3.0]);
        assert_eq!(resolved.state().lp().a().column(2), array![-1.0, 0.0]);
    }

    #[test]
    fn solved_simplex_reoptimizes_after_column_addition() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };

        let (resolved, column) = solved
            .reoptimize_with_added_column(array![1.0, 1.0], -1.0, &mut trace)
            .unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(column, 4);
        assert_eq!(solution.basis_indices, vec![2, 4]);
        assert_eq!(solution.primal, array![0.0, 0.0, 1.0, 0.0, 3.0]);
        assert_eq!(resolved.state().lp().a().column(column), array![1.0, 1.0]);
        assert_eq!(resolved.state().lp().c()[column], -1.0);
    }

    #[test]
    fn solved_simplex_reoptimizes_after_nonbasis_column_removal() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };

        let resolved = solved.reoptimize_without_column(0, &mut trace).unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(solution.basis_indices, vec![2, 0]);
        assert_eq!(solution.primal, array![3.0, 0.0, 4.0]);
        assert_eq!(resolved.state().lp().a().ncols(), 3);
    }

    #[test]
    fn solved_simplex_repairs_basis_after_basis_column_removal() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![1.0, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };

        let resolved = solved.reoptimize_without_column(2, &mut trace).unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(solution.basis_indices, vec![0, 2]);
        assert_eq!(solution.primal, array![4.0, 0.0, 3.0]);
        assert_eq!(resolved.state().lp().a().ncols(), 3);
    }

    #[test]
    fn solved_simplex_keeps_solution_after_satisfied_less_equal_constraint_addition() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![-1.0, -2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };

        let (resolved, slack_column) = solved
            .reoptimize_with_less_equal_constraint(array![1.0, 1.0, 0.0, 0.0], 10.0, &mut trace)
            .unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(slack_column, 4);
        assert_eq!(solution.basis_indices, vec![0, 1, 4]);
        assert_eq!(solution.primal, array![4.0, 3.0, 0.0, 0.0, 3.0]);
        assert_eq!(solution.objective_value, -10.0);
        assert_eq!(solution.iterations, 0);
    }

    #[test]
    fn solved_simplex_reoptimizes_after_violated_less_equal_constraint_addition() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![4.0, 3.0],
            array![-1.0, -2.0, 0.0, 0.0],
        )
        .unwrap();
        let mut trace = NoTrace;
        let reusable =
            primal::solve_reusable(lp, RevisedSimplexOptions::default(), &mut trace).unwrap();
        let primal::ReusableSolveResult::Solved(solved) = reusable else {
            panic!("expected reusable solved state");
        };

        let (resolved, slack_column) = solved
            .reoptimize_with_less_equal_constraint(array![1.0, 1.0, 0.0, 0.0], 5.0, &mut trace)
            .unwrap();

        let SimplexResult::Optimal(solution) = resolved.result() else {
            panic!("expected optimal reoptimized result");
        };
        assert_eq!(slack_column, 4);
        assert_eq!(solution.basis_indices, vec![0, 1, 2]);
        assert_eq!(solution.primal, array![2.0, 3.0, 2.0, 0.0, 0.0]);
        assert_eq!(solution.objective_value, -8.0);
        assert_eq!(resolved.state().lp().a().nrows(), 3);
        assert_eq!(resolved.state().lp().a().ncols(), 5);
    }
}
