use crate::num::Float;
use crate::primitives::partition::*;
use crate::primitives::*;

use rand_core::RngCore;
use thiserror::Error;

/// A floating point type for use with χ² distributions.
pub trait ChiSquaredFloat: Float {
    #[doc(hidden)]
    type P: Partition<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const NORMALIZED_TAIL_POS: Self;
}

impl ChiSquaredFloat for f32 {
    #[doc(hidden)]
    type P = P512<f32>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-2;
    #[doc(hidden)]
    const NORMALIZED_TAIL_POS: Self = 3.25;
}

impl ChiSquaredFloat for f64 {
    #[doc(hidden)]
    type P = P512<f64>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const NORMALIZED_TAIL_POS: Self = 3.25;
}

/// Error type for χ² distribution construction failures.
#[derive(Error, Debug)]
pub enum ChiSquaredError {
    /// The ETF table could not be computed for the provided distribution parameters.
    #[error("could not compute an ETF table for the provided distribution parameters")]
    TabulationFailure,
    /// The number of degrees of freedom should be 2 or greater.
    #[error("the number of degrees of freedom should be 2 or greater")]
    BadDof,
}

/// χ² distribution with more than 2 degree of freedom.
#[derive(Clone)]
pub struct ChiSquared<T: ChiSquaredFloat> {
    inner: DistAnyTailed<
        T::P,
        T,
        UnscaledPdf<T>,
        HorizontalTailEnvelope<T>,
    >,
}

impl<T: ChiSquaredFloat> ChiSquared<T> {
    /// Constructs a χ² distribution with the specified number of degrees of
    /// freedom.
    pub fn new(k: T) -> Result<Self, ChiSquaredError> {
        if k < T::TWO {
            return Err(ChiSquaredError::BadDof);
        }
        // The tail position is determined by keeping the tail sampling
        // probability roughly constant, using the Wilson and Hilferty χ²
        // approximation.
        let normal_variance = T::TWO / (T::from(9.0) * k);
        let normal_mean = T::ONE - normal_variance;
        let normal_tail_pos = normal_mean + T::NORMALIZED_TAIL_POS * normal_variance.sqrt();
        let tail_pos = k * (normal_tail_pos * normal_tail_pos * normal_tail_pos);

        let m = T::ONE_HALF * k - T::ONE;
        let pdf = UnscaledPdf::new(k);
        let dpdf = |x| {
            if x == T::ZERO {
                T::ZERO
            } else {
                (m - T::ONE_HALF * x) * (x.ln() * (m - T::ONE) - T::ONE_HALF * x).exp()
            }
        };
        let init_nodes = util::midpoint_prepartition(&pdf, T::ZERO, tail_pos, 0);
        let mut extrema: &[T] = &[k - T::TWO];
        if k <= T::TWO {
            extrema = &[];
        }
        let table =
            util::newton_tabulation(&pdf, &dpdf, &init_nodes, extrema, T::TOLERANCE, T::ONE, 50)
                .map_err(|_| ChiSquaredError::TabulationFailure)?;
        // Tail envelope, based on an exponential distribution.
        let tail_envelope = HorizontalTailEnvelope::new(k, tail_pos);
        let tail_area = tail_envelope.area();

        Ok(Self {
            inner: DistAnyTailed::new(pdf, &table, tail_envelope, tail_area),
        })
    }
}

impl<T: ChiSquaredFloat> Distribution<T> for ChiSquared<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}

/// Non-normalized χ² probability distribution function with arbitrary
/// location and scale.
#[derive(Copy, Clone, Debug)]
struct UnscaledPdf<T> {
    m: T, // k/2 - 1
}

impl<T: Float> UnscaledPdf<T> {
    fn new(k: T) -> Self {
        Self {
            m: T::ONE_HALF * k - T::ONE,
        }
    }
}

impl<T: Float> UnivariateFn<T> for UnscaledPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        if x == T::ZERO {
            T::ZERO
        } else {
            (x.ln() * self.m - T::ONE_HALF * x).exp()
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct HorizontalTailEnvelope<T> {
    cut_in: T,
    m: T,
    b: T,
    alpha: T,
    beta: T,
}

impl<T: Float> HorizontalTailEnvelope<T> {
    fn new(k: T, cut_in: T) -> Self {
        let m = T::ONE_HALF * k - T::ONE;

        Self {
            cut_in,
            m,
            b: cut_in/(cut_in*T::ONE_HALF - m),
            beta: m / cut_in,
            alpha: (m - m * cut_in.ln()).exp(),
        }
    }
    fn area(&self) -> T {
        let inv_b = T::ONE_HALF - self.beta;
        
        self.b.abs()*(-self.cut_in*inv_b).exp()/self.alpha
    }
}

impl<T: Float> Envelope<T> for HorizontalTailEnvelope<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        let x = self.cut_in - self.b*(T::ONE - T::gen(rng)).ln();
        // In case of a left tail, it is necessary to discard negative `x`
        // values before evaluating `x.ln()`.
        if x<=T::ZERO { 
            return None;
        }
        let p = self.alpha * (self.m * x.ln() - x*self.beta).exp();
        if p > T::gen(rng) {
            Some(x)
        } else {
            None
        }
    }
}
