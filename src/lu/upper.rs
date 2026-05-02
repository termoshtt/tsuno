#[katexit::katexit]
/// Storage for U (upper-triangle or trapezoidal) matrix of LU decomposition.
///
/// $U$ is stored similarly to CSR (Compressed Sparse Row) format, but the non-zero entries are non-contiguous for dynamic update.
///
/// Invariant
/// ----------
/// - The first non-zero entry in each row is the pivot, and the pivot is non-zero.
///
pub struct U {
    arena: Vec<NonZeroEntry>,
    rows: Vec<RowPtr>,
}

impl U {
    pub(crate) fn from_rows(rows: impl IntoIterator<Item = Vec<(usize, f64)>>) -> Self {
        let mut arena = Vec::new();
        let mut row_ptrs = Vec::new();
        for row in rows {
            let offset = arena.len();
            let length = row.len();
            if let Some((_, pivot)) = row.first() {
                assert!(*pivot != 0.0, "U row pivot must be non-zero");
            }
            arena.extend(
                row.into_iter()
                    .map(|(col, value)| NonZeroEntry { value, col }),
            );
            row_ptrs.push(RowPtr { offset, length });
        }
        Self {
            arena,
            rows: row_ptrs,
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = (usize, f64)> + '_> + '_ {
        self.rows.iter().map(|row| {
            self.arena[row.offset..row.offset + row.length]
                .iter()
                .map(|entry| (entry.col, entry.value))
        })
    }
}

/// Pointer of arena for each row in [U].
///
/// Since [U] allows non-zero entries to be non-contiguous, we need to store the offset and length of each row in the arena.
///
struct RowPtr {
    /// Offset of the first non-zero entry in the arena for the row.
    offset: usize,
    /// Number of non-zero entries in the row. `length == 0` means the row is empty, so the matrix is singular.
    length: usize,
}

/// Non-zero entry in with its column index in [U].
struct NonZeroEntry {
    value: f64,
    col: usize,
}
