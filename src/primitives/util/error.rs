
use std::error::Error;
use std::fmt;


/// A type alias for `Result<T, TabulationError>`.
pub type TabulationResult<T> = Result<T, TabulationError>;

/// An error that can occur during a tabulation computation.
#[derive(Debug)]
pub struct TabulationError {
    iteration_count: u32,
}

impl TabulationError {
    pub(super) fn new(iteration_count: u32) -> Self {
        Self { iteration_count }
    }
}

impl fmt::Display for TabulationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "tabulation computation did not converge after {} iterations",
            self.iteration_count
        )
    }
}

impl Error for TabulationError {}
