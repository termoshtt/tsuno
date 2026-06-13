use ndarray::Array1;

use crate::lu::LU;

use super::Basis;

#[derive(Debug)]
pub(super) enum BasisRepresentation {
    Factorized(LU),
    LessEqualSlack {
        base: Box<Basis>,
        basis_row: Array1<f64>,
    },
}

impl BasisRepresentation {
    pub(super) fn rank(&self) -> usize {
        match self {
            BasisRepresentation::Factorized(lu) => lu.row_permutation().len(),
            BasisRepresentation::LessEqualSlack { base, .. } => base.rank() + 1,
        }
    }

    pub(super) fn update_count(&self) -> usize {
        match self {
            BasisRepresentation::Factorized(_) => 0,
            BasisRepresentation::LessEqualSlack { base, .. } => base.update_count(),
        }
    }

    pub(super) fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self {
            BasisRepresentation::Factorized(lu) => lu.solve(rhs),
            BasisRepresentation::LessEqualSlack { base, basis_row } => {
                let base_dimension = base.dimension();
                let base_rhs = Array1::from_iter(rhs.iter().take(base_dimension).copied());
                let base_solution = base.solve(&base_rhs);
                let slack_value = rhs[base_dimension] - basis_row.dot(&base_solution);
                let mut solution = Array1::zeros(base_dimension + 1);
                for (index, &value) in base_solution.iter().enumerate() {
                    solution[index] = value;
                }
                solution[base_dimension] = slack_value;
                solution
            }
        }
    }

    pub(super) fn solve_transposed(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self {
            BasisRepresentation::Factorized(lu) => lu.solve_transposed(rhs),
            BasisRepresentation::LessEqualSlack { base, basis_row } => {
                let base_dimension = base.dimension();
                let slack_rhs = rhs[base_dimension];
                let base_rhs = Array1::from_iter(
                    rhs.iter()
                        .take(base_dimension)
                        .zip(basis_row.iter())
                        .map(|(&value, &row_value)| value - row_value * slack_rhs),
                );
                let base_solution = base.solve_transposed(&base_rhs);
                let mut solution = Array1::zeros(base_dimension + 1);
                for (index, &value) in base_solution.iter().enumerate() {
                    solution[index] = value;
                }
                solution[base_dimension] = slack_rhs;
                solution
            }
        }
    }
}
