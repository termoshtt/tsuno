use crate::simplex::primal;
use crate::simplex::{
    FarkasCertificate, NoTrace, RevisedSimplexOptions, SimplexError, SimplexResult,
    StandardFormError, StandardFormLp,
};

#[katexit::katexit]
/// Irreducible infeasible row subsystem of a standard-form LP.
///
/// This type is deliberately tied to [`StandardFormLp`]. For
///
/// $$
/// A x = b,\qquad x \ge 0,
/// $$
///
/// and a row set $R \subseteq \{0,\ldots,m-1\}$, the corresponding row
/// subsystem is
///
/// $$
/// A_R x = b_R,\qquad x \ge 0.
/// $$
///
/// An IIS is an infeasible row subsystem such that every proper subsystem is
/// feasible. Since this is still below the future high-level LP modeling
/// layer, `rows` contains standard-form row indices, not caller-facing
/// constraint identifiers.
///
/// The `certificate` proves infeasibility of the row subsystem in `rows`,
/// using the same row order as `rows`.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardFormIis {
    pub rows: Vec<usize>,
    pub certificate: FarkasCertificate,
}

#[derive(Clone, Debug, PartialEq)]
/// Result of searching for a standard-form row IIS.
pub enum IisResult {
    Feasible,
    Infeasible(StandardFormIis),
    IterationLimit { rows: Vec<usize> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum IisError {
    Problem(StandardFormError),
    Simplex(SimplexError),
}

impl From<StandardFormError> for IisError {
    fn from(error: StandardFormError) -> Self {
        IisError::Problem(error)
    }
}

impl From<SimplexError> for IisError {
    fn from(error: SimplexError) -> Self {
        IisError::Simplex(error)
    }
}

/// Compute a standard-form row IIS with a deletion filter.
///
/// This first checks whether the full system is infeasible. If it is feasible,
/// [`IisResult::Feasible`] is returned. Otherwise, starting from all rows, the
/// deletion filter tries to remove each row. A row is deleted when the remaining
/// subsystem is still infeasible; it is kept when deletion makes the subsystem
/// feasible.
///
/// This is a correctness-first implementation. It repeatedly solves Phase I
/// feasibility problems and does not try to reuse bases between row deletions.
pub fn deletion_filter_iis(
    lp: &StandardFormLp,
    options: RevisedSimplexOptions,
) -> Result<IisResult, IisError> {
    let mut rows: Vec<_> = (0..lp.a().nrows()).collect();
    match feasibility(lp, &rows, options.clone())? {
        Feasibility::Feasible => return Ok(IisResult::Feasible),
        Feasibility::Infeasible(_) => {}
        Feasibility::IterationLimit => return Ok(IisResult::IterationLimit { rows }),
    }

    let mut position = 0;
    while position < rows.len() {
        let mut candidate_rows = rows.clone();
        candidate_rows.remove(position);

        match feasibility(lp, &candidate_rows, options.clone())? {
            Feasibility::Feasible => {
                position += 1;
            }
            Feasibility::Infeasible(_) => {
                rows = candidate_rows;
            }
            Feasibility::IterationLimit => {
                return Ok(IisResult::IterationLimit {
                    rows: candidate_rows,
                });
            }
        }
    }

    match feasibility(lp, &rows, options)? {
        Feasibility::Infeasible(certificate) => {
            Ok(IisResult::Infeasible(StandardFormIis { rows, certificate }))
        }
        Feasibility::Feasible => Ok(IisResult::Feasible),
        Feasibility::IterationLimit => Ok(IisResult::IterationLimit { rows }),
    }
}

enum Feasibility {
    Feasible,
    Infeasible(FarkasCertificate),
    IterationLimit,
}

fn feasibility(
    lp: &StandardFormLp,
    rows: &[usize],
    options: RevisedSimplexOptions,
) -> Result<Feasibility, IisError> {
    if rows.is_empty() {
        return Ok(Feasibility::Feasible);
    }

    let restricted_lp = lp.row_subsystem(rows)?;
    let mut trace = NoTrace;
    match primal::solve(restricted_lp, options, &mut trace)? {
        SimplexResult::Infeasible(infeasible) => {
            Ok(Feasibility::Infeasible(infeasible.certificate))
        }
        SimplexResult::PhaseOneIterationLimit(_) => Ok(Feasibility::IterationLimit),
        SimplexResult::Optimal(_)
        | SimplexResult::IterationLimit(_)
        | SimplexResult::Unbounded { .. } => Ok(Feasibility::Feasible),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn deletion_filter_iis_returns_feasible_for_feasible_lp() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![1.0, 2.0],
            array![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let result = deletion_filter_iis(&lp, RevisedSimplexOptions::default()).unwrap();

        assert_eq!(result, IisResult::Feasible);
    }

    #[test]
    fn deletion_filter_iis_removes_redundant_rows() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],],
            array![1.0, 2.0, 0.0],
            array![0.0, 0.0, 0.0],
        )
        .unwrap();

        let result = deletion_filter_iis(&lp, RevisedSimplexOptions::default()).unwrap();

        let IisResult::Infeasible(iis) = result else {
            panic!("expected an IIS");
        };
        assert_eq!(iis.rows, vec![0, 1]);

        let subsystem = lp.row_subsystem(&iis.rows).unwrap();
        let verification = iis.certificate.verify(&subsystem, 1.0e-9).unwrap();
        assert!(verification.valid);
        assert_abs_diff_eq!(verification.minimum_column_value, 0.0, epsilon = 1.0e-9);
        assert!(verification.rhs_value < 0.0);
    }

    #[test]
    fn deletion_filter_iis_keeps_every_necessary_row() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [1.0, 0.0]],
            array![1.0, 2.0],
            array![0.0, 0.0],
        )
        .unwrap();

        let result = deletion_filter_iis(&lp, RevisedSimplexOptions::default()).unwrap();

        let IisResult::Infeasible(iis) = result else {
            panic!("expected an IIS");
        };
        assert_eq!(iis.rows, vec![0, 1]);
    }
}
