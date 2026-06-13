use ndarray::{Array1, Array2};
use rand::Rng;

pub(crate) fn diagonally_dominant_matrix(
    size: usize,
    density: f64,
    rng: &mut impl Rng,
) -> Array2<f64> {
    assert!((0.0..=1.0).contains(&density));

    let mut matrix = Array2::zeros((size, size));
    for row in 0..size {
        let mut row_sum = 0.0;
        for col in 0..size {
            if row == col || rng.gen_range(0.0..1.0) >= density {
                continue;
            }
            let value: f64 = rng.gen_range(-1.0..=1.0);
            matrix[(row, col)] = value;
            row_sum += value.abs();
        }
        matrix[(row, row)] = row_sum + 1.0 + rng.gen_range(0.0..1.0);
    }
    matrix
}

pub(crate) fn vector(size: usize, rng: &mut impl Rng) -> Array1<f64> {
    Array1::from_iter((0..size).map(|_| rng.gen_range(-1.0..=1.0)))
}
