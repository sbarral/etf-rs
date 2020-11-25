use crate::num::Float;
use crate::primitives::partition::*;
use crate::primitives::*;

use rand_core::RngCore;
use thiserror::Error;

/// A floating point type for use with Cauchy distributions.
pub trait CauchyFloat: Float {
    #[doc(hidden)]
    type P: Partition<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const TAIL_POS: Self;
}

impl CauchyFloat for f32 {
    #[doc(hidden)]
    type P = P256<f32>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-4;
    #[doc(hidden)]
    const TAIL_POS: Self = 200.0;
}

impl CauchyFloat for f64 {
    #[doc(hidden)]
    type P = P256<f64>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const TAIL_POS: Self = 400.0;
}

/// Error type for Cauchy distribution construction failures.
#[derive(Error, Debug)]
pub enum CauchyError {
    /// The ETF table could not be computed for the provided distribution parameters.
    #[error("could not compute an ETF table for the provided distribution parameters")]
    TabulationFailure,
    /// The provided scale is not strictly positive.
    #[error("the scale should be strictly positive")]
    BadScale,
}

/// Cauchy distribution with arbitrary location and scale.
#[derive(Clone)]
pub struct Cauchy<T: CauchyFloat> {
    inner: DistSymmetricTailed<T::P, T, UnscaledPdf<T>, TailEnvelope<T>>,
}

impl<T: CauchyFloat> Cauchy<T> {
    /// Constructs a Cauchy distribution with the specified location and scale.
    pub fn new(location: T, scale: T) -> Result<Self, CauchyError> {
        if scale <= T::ZERO {
            return Err(CauchyError::BadScale);
        }
        let pdf = UnscaledPdf::new(location, scale);
        let square_inv_scale = T::ONE / (scale * scale);
        let minus_two_square_inv_scale = -T::TWO * square_inv_scale;
        let dpdf = |x| {
            let dx = x - location;

            let minus_dv = minus_two_square_inv_scale * dx;
            let v = T::ONE + square_inv_scale * dx * dx;

            minus_dv / (v * v)
        };

        let tail_position = location + T::TAIL_POS * scale;
        let tail_area = scale * (T::atan(-T::TAIL_POS) + T::ONE_HALF * T::PI);
        let init_nodes = util::midpoint_prepartition(&pdf, location, tail_position, 0);
        let table =
            util::newton_tabulation(&pdf, &dpdf, &init_nodes, &[], T::TOLERANCE, T::ONE, 50)
                .map_err(|_| CauchyError::TabulationFailure)?;
        let tail_func = TailEnvelope::new(location, scale, tail_position);
        Ok(Self {
            inner: DistSymmetricTailed::new(location, pdf, &table, tail_func, tail_area),
        })
    }
}

impl<T: CauchyFloat> Distribution<T> for Cauchy<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}

/// Non-normalized Cauchy probability distribution function with arbitrary
/// location and scale.
#[derive(Copy, Clone, Debug)]
struct UnscaledPdf<T> {
    location: T,
    square_inv_scale: T,
}

impl<T: Float> UnscaledPdf<T> {
    fn new(location: T, scale: T) -> Self {
        Self {
            location,
            square_inv_scale: T::ONE / (scale * scale),
        }
    }
}

impl<T: Float> UnivariateFn<T> for UnscaledPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        let dx = x - self.location;

        T::ONE / (T::ONE + self.square_inv_scale * dx * dx)
    }

    #[inline]
    fn test(&self, x: T, a: T, b: T) -> bool {
        let dx = x - self.location;

        a > b * (T::ONE + self.square_inv_scale * dx * dx)
    }
}

#[derive(Copy, Clone, Debug)]
struct TailEnvelope<T> {
    location: T,
    scale: T,
    a: T,
    b: T,
}

impl<T: Float> TailEnvelope<T> {
    fn new(location: T, scale: T, cut_in: T) -> Self {
        let fmin = T::atan((cut_in - location) / scale) / T::PI + T::ONE_HALF;

        Self {
            location,
            scale,
            a: T::PI * (T::ONE - fmin),
            b: T::PI * (fmin - T::ONE_HALF),
        }
    }
}

impl<T: Float> Envelope<T> for TailEnvelope<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        Some(self.location + self.scale * T::tan(self.a * T::gen(rng) + self.b))
    }
}
