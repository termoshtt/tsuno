use crate::simplex::primal;
use crate::simplex::{
    FarkasCertificate, FarkasVerification, NoTrace, RevisedSimplexOptions, SimplexError,
    SimplexResult, StandardFormError, StandardFormLp,
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
    Infeasible(StandardFormIis),
    IterationLimit { rows: Vec<usize> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum IisError {
    InvalidCertificate(FarkasVerification),
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
    /// Refine this infeasibility certificate into a standard-form row IIS.
    ///
    /// This method treats IIS construction as an additional analysis performed
    /// after infeasibility has already been proved. It first verifies this
    /// certificate against its stored LP; only then does it run a deletion
    /// filter over the standard-form rows.
    ///
    /// For a valid certificate $y$ of
    ///
    /// $$
    /// A x = b,\qquad x \ge 0,
    /// $$
    ///
    /// the full row system is known to be infeasible because
    /// $A^T y \ge 0$ and $b^T y < 0$. The deletion filter then removes a row
    /// whenever the remaining row subsystem is still infeasible. The returned
    /// certificate belongs to the final row subsystem, not necessarily to the
    /// original full system.
    pub fn deletion_filter_iis(
        &self,
        options: RevisedSimplexOptions,
        certificate_tolerance: f64,
    ) -> Result<IisResult, IisError> {
        let verification = self.verify(certificate_tolerance);
        if !verification.valid {
            return Err(IisError::InvalidCertificate(verification));
        }

        let mut rows: Vec<_> = (0..self.lp().a().nrows()).collect();
        let mut position = 0;
        while position < rows.len() {
            let mut candidate_rows = rows.clone();
            candidate_rows.remove(position);

            match feasibility(self.lp(), &candidate_rows, options.clone())? {
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

        match feasibility(self.lp(), &rows, options)? {
            Feasibility::Infeasible(certificate) => {
                Ok(IisResult::Infeasible(StandardFormIis { rows, certificate }))
            }
            Feasibility::Feasible => Err(IisError::InvalidCertificate(verification)),
            Feasibility::IterationLimit => Ok(IisResult::IterationLimit { rows }),
        }
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
    fn deletion_filter_iis_rejects_invalid_certificate() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            array![1.0, 2.0],
            array![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let error = FarkasCertificate::new(lp, array![0.0, 0.0], 1.0e-9).unwrap_err();

        let StandardFormError::InvalidFarkasCertificate(verification) = error else {
            panic!("expected an invalid certificate error");
        };
        assert!(!verification.valid);
    }

    #[test]
    fn deletion_filter_iis_removes_redundant_rows() {
        let lp = StandardFormLp::new(
            array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],],
            array![1.0, 2.0, 0.0],
            array![0.0, 0.0, 0.0],
        )
        .unwrap();
        let certificate =
            FarkasCertificate::new(lp.clone(), array![1.0, -1.0, 0.0], 1.0e-9).unwrap();

        let result = certificate
            .deletion_filter_iis(RevisedSimplexOptions::default(), 1.0e-9)
            .unwrap();

        let IisResult::Infeasible(iis) = result else {
            panic!("expected an IIS");
        };
        assert_eq!(iis.rows, vec![0, 1]);

        let subsystem = lp.row_subsystem(&iis.rows).unwrap();
        assert_eq!(iis.certificate.lp(), &subsystem);
        let verification = iis.certificate.verify(1.0e-9);
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
        let certificate = FarkasCertificate::new(lp, array![1.0, -1.0], 1.0e-9).unwrap();

        let result = certificate
            .deletion_filter_iis(RevisedSimplexOptions::default(), 1.0e-9)
            .unwrap();

        let IisResult::Infeasible(iis) = result else {
            panic!("expected an IIS");
        };
        assert_eq!(iis.rows, vec![0, 1]);
    }
}
