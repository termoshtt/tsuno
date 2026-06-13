use ndarray::Array1;

#[derive(Clone, Debug)]
pub(super) struct BasisEtaUpdate {
    pivot: usize,
    column: Array1<f64>,
}

impl BasisEtaUpdate {
    pub(super) fn new(pivot: usize, column: Array1<f64>) -> Self {
        Self { pivot, column }
    }

    pub(super) fn apply_inverse(&self, vector: &mut Array1<f64>) {
        debug_assert_eq!(vector.len(), self.column.len());

        let pivot_value = self.column[self.pivot];
        let pivot = vector[self.pivot] / pivot_value;
        for index in 0..vector.len() {
            if index != self.pivot {
                vector[index] -= self.column[index] * pivot;
            }
        }
        vector[self.pivot] = pivot;
    }

    pub(super) fn apply_inverse_transposed(&self, vector: &mut Array1<f64>) {
        debug_assert_eq!(vector.len(), self.column.len());

        let pivot_value = self.column[self.pivot];
        let off_pivot_dot = self
            .column
            .iter()
            .zip(vector.iter())
            .enumerate()
            .filter(|(index, _)| *index != self.pivot)
            .map(|(_, (column, value))| column * value)
            .sum::<f64>();
        vector[self.pivot] = (vector[self.pivot] - off_pivot_dot) / pivot_value;
    }
}
