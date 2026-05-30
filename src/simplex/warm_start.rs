use crate::simplex::dual::{DualRevisedSimplex, DualSimplexError};
use crate::simplex::primal::{PrimalSimplexError, RevisedSimplex};
use crate::simplex::{RevisedSimplexOptions, StandardFormLp};

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
pub enum WarmStartError {
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
    match RevisedSimplex::new(lp.clone(), basis_indices.clone(), options.clone()) {
        Ok(simplex) => Ok(WarmStart::Primal(simplex)),
        Err(primal) => match DualRevisedSimplex::new(lp, basis_indices, options) {
            Ok(simplex) => Ok(WarmStart::Dual(simplex)),
            Err(dual) => Err(WarmStartError::NoReusableBasis { primal, dual }),
        },
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

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
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![1.0, 2.0],
            array![0.0, 0.0],
        )
        .unwrap();

        let error = warm_start(lp, vec![0], RevisedSimplexOptions::default()).unwrap_err();

        assert!(matches!(error, WarmStartError::NoReusableBasis { .. }));
    }
}
