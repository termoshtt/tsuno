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
struct NonZeroStructure {
    index: Vec<usize>,
    length: Vec<usize>,
    location: Vec<usize>,
}

impl NonZeroStructure {
    /// Create a row-major non-zero structure from a contiguous list of (row, column, value) entries sorted by row.
    fn row_major(row_sorted: &Vec<(usize, usize, f64)>) -> Self {
        let mut index = Vec::new();
        let mut length = Vec::new();
        let mut location = Vec::new();

        let mut current_row = None;
        for (row, col, _) in row_sorted {
            if Some(*row) != current_row {
                if let Some(r) = current_row {
                    length.push(index.len() - location[r]);
                }
                location.push(index.len());
                current_row = Some(*row);
            }
            index.push(*col);
        }
        if let Some(r) = current_row {
            length.push(index.len() - location[r]);
        }

        // Sanity check: the total number of non-zero entries should match the sum of lengths.
        debug_assert!(index.len() == length.iter().sum::<usize>());
        // Sanity check: the number of rows/columns should match the length of location.
        debug_assert!(location.len() == length.len());

        Self {
            index,
            length,
            location,
        }
    }

    /// Create a column-major non-zero structure from a contiguous list of (row, column, value) entries sorted by column.
    fn col_major(col_sorted: &Vec<(usize, usize, f64)>) -> Self {
        let mut index = Vec::new();
        let mut length = Vec::new();
        let mut location = Vec::new();

        let mut current_col = None;
        for (row, col, _) in col_sorted {
            if Some(*col) != current_col {
                if let Some(c) = current_col {
                    length.push(index.len() - location[c]);
                }
                location.push(index.len());
                current_col = Some(*col);
            }
            index.push(*row);
        }
        if let Some(c) = current_col {
            length.push(index.len() - location[c]);
        }

        // Sanity check: the total number of non-zero entries should match the sum of lengths.
        debug_assert!(index.len() == length.iter().sum::<usize>());
        // Sanity check: the number of rows/columns should match the length of location.
        debug_assert!(location.len() == length.len());

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
    pub fn from_coo_matrix(coo: impl Iterator<Item = (usize, usize, f64)>) -> Self {
        let mut entries = coo.collect::<Vec<_>>();
        entries.sort_by_key(|&(row, col, _)| (row, col));
        let row_major = NonZeroStructure::row_major(&entries);
        entries.sort_by_key(|&(row, col, _)| (col, row));
        let col_major = NonZeroStructure::col_major(&entries);
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

        let structure = NonZeroStructure::row_major(&row_sorted);

        assert_eq!(structure.index, vec![0, 1, 1, 1, 2, 0]);
        assert_eq!(structure.length, vec![2, 1, 2, 1]);
        assert_eq!(structure.location, vec![0, 2, 3, 5]);
    }

    #[test]
    fn row_major_handles_empty_entries() {
        let structure = NonZeroStructure::row_major(&Vec::new());

        assert!(structure.index.is_empty());
        assert!(structure.length.is_empty());
        assert!(structure.location.is_empty());
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

        let structure = NonZeroStructure::col_major(&col_sorted);

        assert_eq!(structure.index, vec![0, 3, 0, 1, 2, 2]);
        assert_eq!(structure.length, vec![2, 3, 1]);
        assert_eq!(structure.location, vec![0, 2, 5]);
    }

    #[test]
    fn col_major_handles_empty_entries() {
        let structure = NonZeroStructure::col_major(&Vec::new());

        assert!(structure.index.is_empty());
        assert!(structure.length.is_empty());
        assert!(structure.location.is_empty());
    }
}
