use crate::num::Float;
use crate::primitives::partition::*;
use crate::primitives::*;

use rand_core::RngCore;
use thiserror::Error;

mod large_shape;
use large_shape::LargeShapeGamma;
mod small_shape;
use small_shape::SmallShapeGamma;

/// A floating point type for use with Γ distributions.
pub trait GammaFloat: Float {
    #[doc(hidden)]
    type P: Partition<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const LARGE_SHAPE_NORMALIZED_TAIL_POS: Self;
    #[doc(hidden)]
    const SMALL_SHAPE_LEFT_TAIL_ENVELOPE_PROBABILITY: Self;
    #[doc(hidden)]
    const SMALL_SHAPE_RIGHT_TAIL_MAX_PROBABILITY: Self;
}

impl GammaFloat for f32 {
    #[doc(hidden)]
    type P = P512<f32>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-1;
    #[doc(hidden)]
    const LARGE_SHAPE_NORMALIZED_TAIL_POS: Self = 3.25;
    #[doc(hidden)]
    const SMALL_SHAPE_LEFT_TAIL_ENVELOPE_PROBABILITY: Self = 0.001;
    #[doc(hidden)]
    const SMALL_SHAPE_RIGHT_TAIL_MAX_PROBABILITY: Self = 0.001;
}

impl GammaFloat for f64 {
    #[doc(hidden)]
    type P = P512<f64>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const LARGE_SHAPE_NORMALIZED_TAIL_POS: Self = 3.25;
    #[doc(hidden)]
    const SMALL_SHAPE_LEFT_TAIL_ENVELOPE_PROBABILITY: Self = 0.001;
    #[doc(hidden)]
    const SMALL_SHAPE_RIGHT_TAIL_MAX_PROBABILITY: Self = 0.001;
}

/// Error type for gamma distribution construction failures.
#[derive(Error, Debug)]
pub enum GammaError {
    /// The ETF table could not be computed for the provided distribution parameters.
    #[error("could not compute an ETF table for the provided distribution parameters")]
    TabulationFailure,
    /// The provided shape parameter is not strictly positive.
    #[error("the shape parameter should be strictly positive")]
    BadShape,
    /// The provided scale parameter is not strictly positive.
    #[error("the scale parameter should be strictly positive")]
    BadScale,
}

/// The gamma distribution.
///
/// The probability density function is:
///
/// ```text
/// f(x) = x^(k - 1) exp(-x / θ) / (Γ(k) θ^k)
/// ```
///
/// where the shape parameter `k` and the scale parameter `θ` are strictly positive.
#[derive(Clone)]
pub struct Gamma<T: GammaFloat> {
    inner: GammaInner<T>,
}
impl<T: GammaFloat> Gamma<T> {
    /// Constructs a gamma distribution with the specified shape and scale.
    pub fn new(shape: T, scale: T) -> Result<Self, GammaError> {
        if scale <= T::ZERO {
            return Err(GammaError::BadScale);
        }
        if shape < T::ONE {
            if shape <= T::ZERO {
                return Err(GammaError::BadShape);
            }
            Ok(Self {
                inner: GammaInner::SmallShape(SmallShapeGamma::new(shape, scale)?),
            })
        } else {
            Ok(Self {
                inner: GammaInner::LargeShape(LargeShapeGamma::new(shape, scale)?),
            })
        }
    }
}
impl<T: GammaFloat> Distribution<T> for Gamma<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        match &self.inner {
            GammaInner::LargeShape(f) => f.sample(rng),
            GammaInner::SmallShape(f) => f.sample(rng),
        }
    }
}

#[derive(Clone)]
enum GammaInner<T: GammaFloat> {
    LargeShape(LargeShapeGamma<T>),
    SmallShape(SmallShapeGamma<T>),
}
