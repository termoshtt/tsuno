//! Initial factorization of a COO matrix into [LU].

use crate::*;

/// Non-zero structure of the matrix with non-contiguous storage without values.
///
/// For matrix:
///
/// ```text
/// A = [[ a     f ],
///      [ b c d   ],
///      [     e   ]]
/// ```
///
/// The non-zero structure can be represented with row-major storage:
///
/// ```text
///             ↓ May lack contiguity in storage within rows
///       a b c - d e f
/// row | 0 0 1 - 2 2 3
/// col | 0 1 1 - 1 2 0  <- This part is stored as `index`
///       |   |   |   ^ Row 3 starts at 6, length = 1
///       |   |   ^-- Row 2 starts at 4, length = 2
///       |   ^ Row 1 starts at 2, length = 1
///       ^-- Row 0 starts at 0, length = 2
/// ```
///
/// This will be stored as follows:
///
/// ```text
/// index = [0 1 1 - 1 2 0]
/// length = [2 1 2 1]
/// location = [0 2 4 6]  <- cannot be derived from `length` due to non-contiguity
/// ```
///
/// This is same for column-major storage, but with rows and columns swapped.
///
/// Invariant
/// ----------
/// - `length.len() == location.len() > 0`
/// - `location` is sorted in ascending order, and `location[0] == 0`
///
struct NonZeroStructure {
    index: Vec<usize>,
    length: Vec<usize>,
    location: Vec<usize>,
}

impl NonZeroStructure {
    /// Create a row-major non-zero structure from a contiguous list of (row, column, value) entries sorted by row.
    fn row_major(nrows: usize, row_sorted: &[(usize, usize, f64)]) -> Self {
        assert!(nrows > 0, "number of rows must be positive");

        let mut index = Vec::new();
        let mut length = Vec::with_capacity(nrows);
        let mut location = Vec::with_capacity(nrows);

        let mut entries = row_sorted.iter().peekable();
        for row in 0..nrows {
            location.push(index.len());
            while let Some(&(entry_row, col, _)) = entries.peek() {
                if *entry_row != row {
                    break;
                }
                index.push(*col);
                entries.next();
            }
            length.push(index.len() - location[row]);
        }
        debug_assert!(entries.peek().is_none());

        // Sanity check: the total number of non-zero entries should match the sum of lengths.
        debug_assert!(index.len() == length.iter().sum::<usize>());
        // Sanity check: the number of rows/columns should match the length of location.
        debug_assert!(location.len() == nrows);
        debug_assert!(length.len() == nrows);

        Self {
            index,
            length,
            location,
        }
    }

    /// Create a column-major non-zero structure from a contiguous list of (row, column, value) entries sorted by column.
    fn col_major(ncols: usize, col_sorted: &[(usize, usize, f64)]) -> Self {
        assert!(ncols > 0, "number of columns must be positive");

        let mut index = Vec::new();
        let mut length = Vec::with_capacity(ncols);
        let mut location = Vec::with_capacity(ncols);

        let mut entries = col_sorted.iter().peekable();
        for col in 0..ncols {
            location.push(index.len());
            while let Some(&(row, entry_col, _)) = entries.peek() {
                if *entry_col != col {
                    break;
                }
                index.push(*row);
                entries.next();
            }
            length.push(index.len() - location[col]);
        }
        debug_assert!(entries.peek().is_none());

        // Sanity check: the total number of non-zero entries should match the sum of lengths.
        debug_assert!(index.len() == length.iter().sum::<usize>());
        // Sanity check: the number of rows/columns should match the length of location.
        debug_assert!(location.len() == ncols);
        debug_assert!(length.len() == ncols);

        Self {
            index,
            length,
            location,
        }
    }
}

/// Workspace for initial factorization of a COO matrix into [LU].
pub struct Worker {
    /// Current step of the factorization process, $k$ in the paper.
    step: usize,

    /// Non-zero values in column-major order, aligned with `col_major` in [NonZeroStructure].
    non_zeros: Vec<f64>,
    col_major: NonZeroStructure,
    row_major: NonZeroStructure,
}

impl Worker {
    pub fn from_coo_matrix(
        nrows: usize,
        ncols: usize,
        coo: impl Iterator<Item = (usize, usize, f64)>,
    ) -> Self {
        assert!(nrows > 0, "number of rows must be positive");
        assert!(ncols > 0, "number of columns must be positive");

        let mut entries = coo.collect::<Vec<_>>();
        for &(row, col, _) in &entries {
            assert!(
                row < nrows,
                "row index {row} is out of bounds for {nrows} rows"
            );
            assert!(
                col < ncols,
                "column index {col} is out of bounds for {ncols} columns"
            );
        }

        entries.sort_by_key(|&(row, col, _)| (row, col));
        let row_major = NonZeroStructure::row_major(nrows, &entries);
        entries.sort_by_key(|&(row, col, _)| (col, row));
        let col_major = NonZeroStructure::col_major(ncols, &entries);
        let non_zeros = entries.into_iter().map(|(_, _, value)| value).collect();
        Self {
            step: 0,
            non_zeros,
            col_major,
            row_major,
        }
    }

    /// Perform the unit factorization step for the current `step`.
    fn unit_factorize(&mut self) {
        todo!()
    }

    pub fn factorize(self) -> LU {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_builds_index_length_and_location() {
        let row_sorted = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 1, 3.0),
            (2, 1, 4.0),
            (2, 2, 5.0),
            (3, 0, 6.0),
        ];

        let structure = NonZeroStructure::row_major(4, &row_sorted);

        assert_eq!(structure.index, vec![0, 1, 1, 1, 2, 0]);
        assert_eq!(structure.length, vec![2, 1, 2, 1]);
        assert_eq!(structure.location, vec![0, 2, 3, 5]);
    }

    #[test]
    #[should_panic(expected = "number of rows must be positive")]
    fn row_major_rejects_zero_rows() {
        NonZeroStructure::row_major(0, &Vec::new());
    }

    #[test]
    fn row_major_represents_empty_rows() {
        let row_sorted = vec![(1, 2, 1.0), (3, 0, 2.0)];

        let structure = NonZeroStructure::row_major(5, &row_sorted);

        assert_eq!(structure.index, vec![2, 0]);
        assert_eq!(structure.length, vec![0, 1, 0, 1, 0]);
        assert_eq!(structure.location, vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn col_major_builds_index_length_and_location() {
        let col_sorted = vec![
            (0, 0, 1.0),
            (3, 0, 6.0),
            (0, 1, 2.0),
            (1, 1, 3.0),
            (2, 1, 4.0),
            (2, 2, 5.0),
        ];

        let structure = NonZeroStructure::col_major(3, &col_sorted);

        assert_eq!(structure.index, vec![0, 3, 0, 1, 2, 2]);
        assert_eq!(structure.length, vec![2, 3, 1]);
        assert_eq!(structure.location, vec![0, 2, 5]);
    }

    #[test]
    #[should_panic(expected = "number of columns must be positive")]
    fn col_major_rejects_zero_columns() {
        NonZeroStructure::col_major(0, &Vec::new());
    }

    #[test]
    fn col_major_represents_empty_columns() {
        let col_sorted = vec![(1, 1, 1.0), (0, 3, 2.0)];

        let structure = NonZeroStructure::col_major(5, &col_sorted);

        assert_eq!(structure.index, vec![1, 0]);
        assert_eq!(structure.length, vec![0, 1, 0, 1, 0]);
        assert_eq!(structure.location, vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn from_coo_matrix_builds_structures_with_empty_rows_and_columns() {
        let worker = Worker::from_coo_matrix(4, 5, vec![(1, 3, 10.0), (3, 1, 20.0)].into_iter());

        assert_eq!(worker.row_major.index, vec![3, 1]);
        assert_eq!(worker.row_major.length, vec![0, 1, 0, 1]);
        assert_eq!(worker.row_major.location, vec![0, 0, 1, 1]);
        assert_eq!(worker.col_major.index, vec![3, 1]);
        assert_eq!(worker.col_major.length, vec![0, 1, 0, 1, 0]);
        assert_eq!(worker.col_major.location, vec![0, 0, 1, 1, 2]);
        assert_eq!(worker.non_zeros, vec![20.0, 10.0]);
    }

    #[test]
    #[should_panic(expected = "number of rows must be positive")]
    fn from_coo_matrix_rejects_zero_rows() {
        Worker::from_coo_matrix(0, 1, Vec::new().into_iter());
    }

    #[test]
    #[should_panic(expected = "number of columns must be positive")]
    fn from_coo_matrix_rejects_zero_columns() {
        Worker::from_coo_matrix(1, 0, Vec::new().into_iter());
    }
}
