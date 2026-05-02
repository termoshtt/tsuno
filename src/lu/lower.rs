use std::fmt;

#[katexit::katexit]
/// Storage for the nominally lower-triangle matrix $L$
///
/// $L$ matrix is represented as a product of unit triangle matrices $L = M_0 M_1 \cdots$,
/// where each unit triangle matrix is
///
/// $$
/// M_k = 1 - \mu_k |r_k\rangle \langle c_k|, \quad r_k \neq c_k
/// $$
///
/// Note that $r_k$ and $c_k$ are just not equal, but they can be in any order.
/// This means that $L$ is not necessarily lower-triangular.
///
#[derive(Debug)]
pub struct L {
    units: Vec<UnitTriangle>,
}

impl L {
    pub(crate) fn from_units(units: impl IntoIterator<Item = UnitTriangle>) -> Self {
        Self {
            units: units.into_iter().collect(),
        }
    }

    pub fn units(&self) -> impl Iterator<Item = (f64, usize, usize)> + '_ {
        self.units.iter().map(|unit| (unit.mu, unit.row, unit.col))
    }
}

impl fmt::Display for L {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L = ")?;
        if self.units.is_empty() {
            write!(f, "1")?;
            return Ok(());
        }
        for unit in &self.units {
            write!(f, "({})", unit)?;
        }
        Ok(())
    }
}

#[katexit::katexit]
/// Unit triangle matrix in the product representation of [L].
///
/// $$
/// M = 1 - \mu |r\rangle \langle c|, \quad r \neq c
/// $$
///
/// Invariant
/// ---------
/// - $\mu \neq 0$
/// - $r \neq c$
///
#[derive(Debug)]
pub(crate) struct UnitTriangle {
    mu: f64,
    col: usize,
    row: usize,
}

impl UnitTriangle {
    pub(crate) fn new(mu: f64, row: usize, col: usize) -> Self {
        assert!(mu != 0.0, "unit triangle multiplier must be non-zero");
        assert_ne!(row, col, "unit triangle row and column must differ");
        Self { mu, row, col }
    }
}

impl fmt::Display for UnitTriangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "1 - {} |{}><{}|", self.mu, self.row, self.col)
    }
}
