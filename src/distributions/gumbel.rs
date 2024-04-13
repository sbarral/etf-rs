use crate::num::Float;
use crate::primitives::partition::*;
use crate::primitives::*;

use rand_core::RngCore;
use thiserror::Error;

/// A floating point type for use with Gumbel distributions.
pub trait GumbelFloat: Float {
    #[doc(hidden)]
    type P: Partition<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const LEFT_TAIL_POS: Self;
    #[doc(hidden)]
    const RIGHT_TAIL_POS: Self;
}

impl GumbelFloat for f32 {
    #[doc(hidden)]
    type P = P256<f32>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-4;
    #[doc(hidden)]
    const LEFT_TAIL_POS: Self = -1.7; // use -1.6 for P128
    #[doc(hidden)]
    const RIGHT_TAIL_POS: Self = 5.5; // use 5.0 for P128
}

impl GumbelFloat for f64 {
    #[doc(hidden)]
    type P = P256<f64>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const LEFT_TAIL_POS: Self = -1.7; // use -1.6 for P128
    #[doc(hidden)]
    const RIGHT_TAIL_POS: Self = 5.5; // use 5.0 for P128
}

/// Error type for Gumbel distribution construction failures.
#[derive(Error, Debug)]
pub enum GumbelError {
    /// The ETF table could not be computed for the provided distribution parameters.
    #[error("could not compute an ETF table for the provided distribution parameters")]
    TabulationFailure,
    /// The provided scale parameter is not strictly positive.
    #[error("the scale parameter should be strictly positive")]
    BadScale,
}

/// The Gumbel distribution.
///
/// The probability density function is:
///
/// ```text
/// f(x) = exp(-(z + exp(-z))) / β
/// ```
/// where:
/// ```
/// z = (x - μ) / β
/// ```
///
/// where `μ` is the location parameter and where the scale parameter `β` is
/// strictly positive.
#[derive(Clone)]
pub struct Gumbel<T: GumbelFloat> {
    inner: DistAnyTailed<T::P, T, UnscaledPdf<T>, Tail<T>>,
}

impl<T: GumbelFloat> Gumbel<T> {
    /// Constructs a Gumbel distribution with the specified location and scale.
    pub fn new(location: T, scale: T) -> Result<Self, GumbelError> {
        if scale <= T::ZERO {
            return Err(GumbelError::BadScale);
        }
        let pdf = UnscaledPdf::new(location, scale);
        let inv_scale = T::ONE / scale;
        let dpdf = |x| {
            let minus_z = (location - x) * inv_scale;
            let exp_minus_z = T::exp(minus_z);

            T::exp(minus_z - exp_minus_z) * (exp_minus_z - T::ONE) * inv_scale
        };

        let left_tail_position = location + T::LEFT_TAIL_POS * scale;
        let right_tail_position = location + T::RIGHT_TAIL_POS * scale;

        let init_nodes =
            util::midpoint_prepartition(&pdf, left_tail_position, right_tail_position, 0);
        let table = util::newton_tabulation(
            &pdf,
            &dpdf,
            &init_nodes,
            &[location],
            T::TOLERANCE,
            T::ONE,
            50,
        )
        .map_err(|_| GumbelError::TabulationFailure)?;
        let (tail_func, tail_area) = Tail::new_with_area(location, scale);

        Ok(Self {
            inner: DistAnyTailed::new(pdf, &table, tail_func, tail_area),
        })
    }
}

impl<T: GumbelFloat> Distribution<T> for Gumbel<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}

/// Non-normalized Gumbel probability distribution function with arbitrary
/// location and scale.
#[derive(Copy, Clone, Debug)]
struct UnscaledPdf<T> {
    location: T,
    inv_scale: T,
}

impl<T: Float> UnscaledPdf<T> {
    fn new(location: T, scale: T) -> Self {
        Self {
            location,
            inv_scale: T::ONE / scale,
        }
    }
}

impl<T: Float> UnivariateFn<T> for UnscaledPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        let minus_z = (self.location - x) * self.inv_scale;

        T::exp(minus_z - T::exp(minus_z))
    }
}

#[derive(Copy, Clone, Debug)]
struct Tail<T> {
    location: T,
    scale: T,
    a_left: T,
    a_right: T,
    rt: T,
}

impl<T: GumbelFloat> Tail<T> {
    fn new_with_area(location: T, scale: T) -> (Self, T) {
        let cdf = |z: T| T::exp(-T::exp(-z));

        let wl = cdf(T::LEFT_TAIL_POS);
        let wr = T::ONE - cdf(T::RIGHT_TAIL_POS);
        let rt = wl / (wl + wr);
        let a_left = wl / rt;
        let a_right = wr / (T::ONE - rt);

        let tail = Self {
            location,
            scale,
            a_left,
            a_right,
            rt,
        };
        let area = (wl + wr) * scale;

        (tail, area)
    }
}

impl<T: Float> TryDistribution<T> for Tail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        let r = T::gen(rng);
        let p = if r < self.rt {
            r * self.a_left
        } else {
            T::ONE - (T::ONE - r) * self.a_right
        };

        Some(self.location - self.scale * T::ln(-T::ln(p)))
    }
}
