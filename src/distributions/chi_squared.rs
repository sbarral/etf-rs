use crate::primitives::Distribution;

use rand_core::RngCore;
use thiserror::Error;

use super::gamma::{Gamma, GammaError, GammaFloat};

/// A floating point type for use with χ² distributions.
pub trait ChiSquaredFloat: GammaFloat {}

impl ChiSquaredFloat for f32 {}

impl ChiSquaredFloat for f64 {}

/// Error type for χ² distribution construction failures.
#[derive(Error, Debug)]
pub enum ChiSquaredError {
    /// The ETF table could not be computed for the provided distribution parameters.
    #[error("could not compute an ETF table for the provided distribution parameters")]
    TabulationFailure,
    /// The number of degrees of freedom is not strictly positive.
    #[error("the number of degrees of freedom should be strictly positive")]
    BadDof,
}

/// The χ² distribution.
/// 
/// The probability density function is:
///
/// ```text
/// f(x) = x^(k / 2 - 1) exp(-x / 2) / (Γ(k / 2) 2^(k / 2))
/// ```
///
/// where the number of degrees of freedom `k` is strictly positive.
#[derive(Clone)]
pub struct ChiSquared<T: ChiSquaredFloat> {
    inner: Gamma<T>,
}

impl<T: ChiSquaredFloat> ChiSquared<T> {
    /// Constructs a χ² distribution with the specified number of degrees of
    /// freedom.
    pub fn new(k: T) -> Result<Self, ChiSquaredError> {
        match Gamma::new(T::ONE_HALF * k, T::TWO) {
            Ok(inner) => Ok(Self { inner }),
            Err(GammaError::TabulationFailure) => Err(ChiSquaredError::TabulationFailure),
            Err(GammaError::BadShape) => Err(ChiSquaredError::BadDof),
            Err(GammaError::BadScale) => unreachable!(),
        }
    }
}

impl<T: ChiSquaredFloat> Distribution<T> for ChiSquared<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}
