use crate::simplex::primal;
use crate::simplex::{
    FarkasCertificate, NoTrace, RevisedSimplexOptions, SimplexError, SimplexResult,
    StandardFormError, StandardFormLp,
};
use ndarray::Array1;

#[derive(Clone, Debug, PartialEq)]
pub enum IisError {
    IterationLimit { rows: Vec<usize> },
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

impl FarkasCertificate {
    #[katexit::katexit]
    /// Simplify this certificate with a deletion filter.
    ///
    /// IIS construction is represented as a transformation from one Farkas
    /// certificate to another certificate for the same stored LP. For
    ///
    /// $$
    /// A x = b,\qquad x \ge 0,
    /// $$
    ///
    /// a certificate multiplier $y$ proves infeasibility by
    ///
    /// $$
    /// A^T y \ge 0,\qquad b^T y < 0.
    /// $$
    ///
    /// The deletion filter tries to remove rows from the support of such a
    /// proof. If the remaining row subsystem is still infeasible, the row is
    /// deleted. The certificate obtained for the reduced subsystem is then
    /// lifted back to the original LP by assigning multiplier value zero to
    /// deleted rows. Therefore the returned certificate still certifies the
    /// same LP as `self`, but its multiplier support is a smaller infeasible
    /// row subsystem.
    pub fn deletion_filter(
        &self,
        options: RevisedSimplexOptions,
    ) -> Result<FarkasCertificate, IisError> {
        let certificate_tolerance = options.pivot_tolerance;
        let mut rows: Vec<_> = (0..self.lp().a().nrows()).collect();
        let mut reduced_certificate = None;
        let mut position = 0;
        while position < rows.len() {
            let mut candidate_rows = rows.clone();
            candidate_rows.remove(position);

            if candidate_rows.is_empty() {
                position += 1;
                continue;
            }

            match solve_row_subsystem(self.lp(), &candidate_rows, options.clone())? {
                SimplexResult::Optimal(_)
                | SimplexResult::IterationLimit(_)
                | SimplexResult::Unbounded { .. } => {
                    position += 1;
                }
                SimplexResult::Infeasible(infeasible) => {
                    rows = candidate_rows;
                    reduced_certificate = Some(infeasible.certificate);
                }
                SimplexResult::PhaseOneIterationLimit(_) => {
                    return Err(IisError::IterationLimit {
                        rows: candidate_rows,
                    });
                }
            }
        }

        if let Some(certificate) = reduced_certificate {
            lift_certificate(self.lp(), &rows, certificate, certificate_tolerance)
        } else {
            Ok(self.clone())
        }
    }
}

fn solve_row_subsystem(
    lp: &StandardFormLp,
    rows: &[usize],
    options: RevisedSimplexOptions,
) -> Result<SimplexResult, IisError> {
    debug_assert!(!rows.is_empty());
    let restricted_lp = lp.row_subsystem(rows)?;
    let mut trace = NoTrace;
    Ok(primal::solve(restricted_lp, options, &mut trace)?)
}

fn lift_certificate(
    lp: &StandardFormLp,
    rows: &[usize],
    certificate: FarkasCertificate,
    tolerance: f64,
) -> Result<FarkasCertificate, IisError> {
    let mut multiplier = Array1::zeros(lp.a().nrows());
    for (&source_row, &value) in rows.iter().zip(certificate.multiplier()) {
        multiplier[source_row] = value;
    }
    FarkasCertificate::new(lp.clone(), multiplier, tolerance).map_err(IisError::Problem)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn farkas_certificate_rejects_invalid_multiplier() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![1.0, 2.0],
            array![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let error = FarkasCertificate::new(lp, array![0.0, 0.0], 1.0e-9).unwrap_err();

        let StandardFormError::InvalidFarkasCertificate {
            minimum_column_value,
            rhs_value,
        } = error
        else {
            panic!("expected an invalid certificate error");
        };
        assert_eq!(minimum_column_value, 0.0);
        assert_eq!(rhs_value, 0.0);
    }

    #[test]
    fn deletion_filter_removes_redundant_rows_from_certificate_support() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],],
            array![1.0, 2.0, 0.0],
            array![0.0, 0.0, 0.0],
        )
        .unwrap();
        let certificate =
            FarkasCertificate::new(lp.clone(), array![1.0, -1.0, 1.0], 1.0e-9).unwrap();

        let simplified = certificate
            .deletion_filter(RevisedSimplexOptions::default())
            .unwrap();

        assert_abs_diff_eq!(
            simplified.multiplier(),
            &array![1.0, -1.0, 0.0],
            epsilon = 1.0e-9
        );
        assert_eq!(simplified.support(1.0e-9), vec![0, 1]);
        assert_eq!(simplified.lp(), &lp);
    }

    #[test]
    fn deletion_filter_keeps_every_necessary_row_in_support() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0], [1.0, 0.0]],
            array![1.0, 2.0],
            array![0.0, 0.0],
        )
        .unwrap();
        let certificate = FarkasCertificate::new(lp, array![1.0, -1.0], 1.0e-9).unwrap();

        let simplified = certificate
            .deletion_filter(RevisedSimplexOptions::default())
            .unwrap();

        assert_eq!(simplified.support(1.0e-9), vec![0, 1]);
    }
}
