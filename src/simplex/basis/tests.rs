use approx::assert_abs_diff_eq;
use ndarray::array;

use super::*;

#[test]
fn basis_builds_matrix_and_solves() {
    let matrix = array![
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [5.0, 0.0, 0.0, 6.0],
    ];
    let basis = Basis::new(&matrix, vec![1, 2, 3]).unwrap();
    let expected_solution = array![2.0, 3.0, 5.0];
    let basis_matrix = array![[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 6.0]];
    let rhs = basis_matrix.dot(&expected_solution);

    let solution = basis.solve(&rhs);

    assert_eq!(basis.indices(), &[1, 2, 3]);
    assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
}

#[test]
fn basis_solves_transposed_system() {
    let matrix = array![
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [5.0, 0.0, 0.0, 6.0],
    ];
    let basis = Basis::new(&matrix, vec![1, 2, 3]).unwrap();
    let expected_solution = array![2.0, 3.0, 5.0];
    let basis_matrix = array![[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 6.0]];
    let rhs = basis_matrix.t().dot(&expected_solution);

    let solution = basis.solve_transposed(&rhs);

    assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
}

#[test]
fn basis_replaces_column_and_updates_indices() {
    let matrix = array![
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [5.0, 0.0, 0.0, 6.0],
    ];
    let mut basis = Basis::new(&matrix, vec![0, 2, 3]).unwrap();
    let expected_solution = array![2.0, 3.0, 5.0];
    let mut updated_basis = array![[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]];

    basis
        .replace_column(0, 1, &matrix.column(1).to_owned())
        .unwrap();
    updated_basis.column_mut(0).assign(&matrix.column(1));
    let rhs = updated_basis.dot(&expected_solution);
    let solution = basis.solve(&rhs);

    assert_eq!(basis.indices(), &[1, 2, 3]);
    assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    assert!(basis.should_refactor(1));
}

#[test]
fn basis_applies_multiple_column_replacements_to_solve_and_transposed_solve() {
    let matrix = array![[2.0, 0.0, 1.0], [4.0, 3.0, 0.0], [0.0, 5.0, 6.0]];
    let first_replacement = array![7.0, 8.0, 9.0];
    let second_replacement = array![3.0, 1.0, 4.0];
    let expected_solution = array![1.0, 2.0, 5.0];
    let mut expected_basis = matrix.clone();
    let mut basis = Basis::new(&matrix, vec![0, 1, 2]).unwrap();

    basis.replace_column(1, 3, &first_replacement).unwrap();
    expected_basis.column_mut(1).assign(&first_replacement);
    basis.replace_column(0, 4, &second_replacement).unwrap();
    expected_basis.column_mut(0).assign(&second_replacement);
    let rhs = expected_basis.dot(&expected_solution);
    let transposed_rhs = expected_basis.t().dot(&expected_solution);

    let solution = basis.solve(&rhs);
    let transposed_solution = basis.solve_transposed(&transposed_rhs);

    assert_eq!(basis.indices(), &[4, 3, 2]);
    assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    assert_abs_diff_eq!(transposed_solution, expected_solution, epsilon = 1.0e-9);
    assert!(basis.should_refactor(2));
}

#[test]
fn basis_extends_less_equal_slack_and_solves_block_system() {
    let matrix = array![[1.0, 0.0], [0.0, 1.0]];
    let basis = Basis::new(&matrix, vec![0, 1])
        .unwrap()
        .extend_with_less_equal_slack(2, array![2.0, 3.0])
        .unwrap();
    let expected_solution = array![4.0, 5.0, 6.0];
    let basis_matrix = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 3.0, 1.0]];
    let rhs = basis_matrix.dot(&expected_solution);

    let solution = basis.solve(&rhs);
    let transposed_rhs = basis_matrix.t().dot(&expected_solution);
    let transposed_solution = basis.solve_transposed(&transposed_rhs);

    assert_eq!(basis.indices(), &[0, 1, 2]);
    assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    assert_abs_diff_eq!(transposed_solution, expected_solution, epsilon = 1.0e-9);
}

#[test]
fn basis_replaces_column_after_less_equal_slack_extension() {
    let matrix = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    let mut basis = Basis::new(&matrix, vec![0, 1])
        .unwrap()
        .extend_with_less_equal_slack(3, array![2.0, 3.0])
        .unwrap();
    let replacement = array![1.0, 1.0, 4.0];
    let expected_solution = array![2.0, 3.0, 5.0];
    let basis_matrix = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [2.0, 3.0, 4.0]];

    basis.replace_column(2, 2, &replacement).unwrap();
    let rhs = basis_matrix.dot(&expected_solution);
    let solution = basis.solve(&rhs);

    assert_eq!(basis.indices(), &[0, 1, 2]);
    assert_abs_diff_eq!(solution, expected_solution, epsilon = 1.0e-9);
    assert!(basis.should_refactor(1));
}

#[test]
fn basis_remaps_indices_after_nonbasis_column_swap_removal() {
    let matrix = array![
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0],
        [5.0, 0.0, 0.0, 6.0],
    ];
    let basis = Basis::new(&matrix, vec![1, 2, 3]).unwrap();

    let basis = basis.remap_indices_after_swap_remove_column(0, 3).unwrap();

    assert_eq!(basis.indices(), &[1, 2, 0]);
}

#[test]
fn basis_rejects_removing_basis_column() {
    let matrix = array![[1.0, 0.0], [0.0, 1.0]];
    let basis = Basis::new(&matrix, vec![0, 1]).unwrap();

    let error = basis
        .remap_indices_after_swap_remove_column(1, 1)
        .unwrap_err();

    assert_eq!(error, BasisError::CannotRemoveBasisColumn { column: 1 });
}

#[test]
fn basis_rejects_wrong_number_of_indices() {
    let matrix = array![[1.0, 0.0], [0.0, 1.0]];

    let error = Basis::new(&matrix, vec![0]).unwrap_err();

    assert_eq!(
        error,
        BasisError::BasisSizeMismatch {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn basis_rejects_more_rows_than_columns() {
    let matrix = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

    let error = Basis::new(&matrix, vec![0, 1, 0]).unwrap_err();

    assert_eq!(error, BasisError::TooFewColumns { nrows: 3, ncols: 2 });
}

#[test]
fn basis_rejects_out_of_bounds_column() {
    let matrix = array![[1.0, 0.0], [0.0, 1.0]];

    let error = Basis::new(&matrix, vec![0, 2]).unwrap_err();

    assert_eq!(
        error,
        BasisError::ColumnOutOfBounds {
            column: 2,
            ncols: 2
        }
    );
}
