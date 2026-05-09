use std::fmt;

use ndarray::Array1;

use super::{LeavingColumn, SimplexStep};
use crate::simplex::PricedColumn;

pub trait SimplexTrace {
    fn phase_started(&mut self, _phase: SimplexTracePhase) {}

    fn step_started(&mut self, _iteration: usize, _basis: &[usize]) {}

    fn step_completed(&mut self, event: SimplexTraceEvent<'_>);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimplexTracePhase {
    PhaseOne,
    PhaseTwo,
}

#[derive(Clone, Copy, Debug)]
pub struct NoTrace;

impl SimplexTrace for NoTrace {
    fn step_completed(&mut self, _event: SimplexTraceEvent<'_>) {}
}

#[derive(Clone, Debug)]
pub struct SimplexTraceEvent<'a> {
    pub iteration: usize,
    pub step: &'a SimplexStep,
    pub basis_after: &'a [usize],
}

/// Trace collector that stores every revised simplex step.
///
/// [`NoTrace`] is the zero-storage trace implementation for ordinary solves.
/// This type is the structured counterpart: it records each step so callers can
/// inspect the path taken by the simplex method, or render it via
/// [`fmt::Display`].
#[derive(Clone, Debug, Default)]
pub struct FullTrace {
    current_phase: Option<SimplexTracePhase>,
    pending_basis_before: Option<Vec<usize>>,
    steps: Vec<FullTraceStep>,
}

/// One recorded revised simplex iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct FullTraceStep {
    pub phase: Option<SimplexTracePhase>,
    pub iteration: usize,
    pub basis_before: Vec<usize>,
    pub outcome: FullTraceOutcome,
    pub basis_after: Vec<usize>,
}

/// Recorded outcome of one revised simplex iteration.
#[derive(Clone, Debug, PartialEq)]
pub enum FullTraceOutcome {
    Optimal,
    Unbounded {
        entering: PricedColumn,
        direction: Array1<f64>,
    },
    Pivoted {
        entering: PricedColumn,
        leaving: LeavingColumn,
        direction: Array1<f64>,
    },
}

impl SimplexTrace for FullTrace {
    fn phase_started(&mut self, phase: SimplexTracePhase) {
        self.current_phase = Some(phase);
    }

    fn step_started(&mut self, _iteration: usize, basis: &[usize]) {
        self.pending_basis_before = Some(basis.to_vec());
    }

    fn step_completed(&mut self, event: SimplexTraceEvent<'_>) {
        let basis_before = self.pending_basis_before.take().unwrap();
        self.steps.push(FullTraceStep {
            phase: self.current_phase,
            iteration: event.iteration,
            basis_before,
            outcome: FullTraceOutcome::from(event.step),
            basis_after: event.basis_after.to_vec(),
        });
    }
}

impl FullTrace {
    pub fn steps(&self) -> &[FullTraceStep] {
        &self.steps
    }
}

impl From<&SimplexStep> for FullTraceOutcome {
    fn from(step: &SimplexStep) -> Self {
        match step {
            SimplexStep::Optimal => FullTraceOutcome::Optimal,
            SimplexStep::Unbounded {
                entering,
                direction,
            } => FullTraceOutcome::Unbounded {
                entering: entering.clone(),
                direction: direction.clone(),
            },
            SimplexStep::Pivoted {
                entering,
                leaving,
                direction,
            } => FullTraceOutcome::Pivoted {
                entering: entering.clone(),
                leaving: leaving.clone(),
                direction: direction.clone(),
            },
        }
    }
}

impl fmt::Display for FullTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, step) in self.steps.iter().enumerate() {
            if index > 0 {
                writeln!(f)?;
                writeln!(f)?;
            }
            write!(f, "{step}")?;
        }
        Ok(())
    }
}

impl fmt::Display for FullTraceStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(phase) = self.phase {
            writeln!(f, "phase: {phase}")?;
        }
        writeln!(f, "iteration {}", self.iteration)?;
        writeln!(f, "basis before: {:?}", self.basis_before)?;
        write!(f, "{}", self.outcome)?;
        write!(f, "basis after: {:?}", self.basis_after)
    }
}

impl fmt::Display for SimplexTracePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimplexTracePhase::PhaseOne => write!(f, "phase one"),
            SimplexTracePhase::PhaseTwo => write!(f, "phase two"),
        }
    }
}

impl fmt::Display for FullTraceOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FullTraceOutcome::Optimal => writeln!(f, "outcome: optimal"),
            FullTraceOutcome::Unbounded {
                entering,
                direction,
            } => {
                writeln!(f, "outcome: unbounded")?;
                writeln!(
                    f,
                    "entering column: {} (reduced_cost: {})",
                    entering.column,
                    format_number(entering.reduced_cost)
                )?;
                writeln!(f, "direction: {}", format_array(direction))
            }
            FullTraceOutcome::Pivoted {
                entering,
                leaving,
                direction,
            } => {
                writeln!(f, "outcome: pivoted")?;
                writeln!(
                    f,
                    "entering column: {} (reduced_cost: {})",
                    entering.column,
                    format_number(entering.reduced_cost)
                )?;
                writeln!(
                    f,
                    "leaving column: {} (step_length: {})",
                    leaving.column,
                    format_number(leaving.step_length)
                )?;
                writeln!(f, "direction: {}", format_array(direction))
            }
        }
    }
}

fn format_array(values: &Array1<f64>) -> String {
    let values = values
        .iter()
        .map(|&value| format_number(value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn format_number(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}
