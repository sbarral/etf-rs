use crate::num::Float;
use crate::primitives::partition::*;
use crate::primitives::*;

use rand_core::RngCore;
use thiserror::Error;

/// A floating point type for use with Γ distributions.
pub trait GammaFloat: Float {
    #[doc(hidden)]
    type P: Partition<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const NORMALIZED_TAIL_POS: Self;
}

impl GammaFloat for f32 {
    #[doc(hidden)]
    type P = P512<f32>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-2;
    #[doc(hidden)]
    const NORMALIZED_TAIL_POS: Self = 3.25;
}

impl GammaFloat for f64 {
    #[doc(hidden)]
    type P = P512<f64>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const NORMALIZED_TAIL_POS: Self = 3.25;
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
        Ok(Self {
            inner: GammaInner::LargeShape(LargeShapeGamma::new(shape, scale)?),
        })
    }
}
impl<T: GammaFloat> Distribution<T> for Gamma<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        match &self.inner {
            GammaInner::LargeShape(f) => f.sample(rng),
        }
    }
}

#[derive(Clone)]
enum GammaInner<T: GammaFloat> {
    LargeShape(LargeShapeGamma<T>),
}

/// Specialized gamma distribution for `k>=1`.
///
/// In order to prevent floating point overflow at large `k`, the implementation
/// uses the following scaled probability density function:
///
/// ```text
/// f(x) = a x^(k - 1) exp(-x / θ)
///      = exp(-x / θ + (k - 1) ln(x) + ln(a))
/// ```
///
/// where constant `a` normalizes the function by its maximum:
///
/// ```text
/// ln(a) = (k - 1)(1 - ln(θ (k - 1)))
/// ```
///
/// The position of the right tail (and, for large `k`, of the left tail) is
/// transformed from the tail position of a normal distribution via the
/// Wilson-Hilferty approximation.
#[derive(Clone)]
struct LargeShapeGamma<T: GammaFloat> {
    inner: DistAnyTailed<T::P, T, LargeShapeUnscaledPdf<T>, LargeShapeTail<T>>,
}
impl<T: GammaFloat> LargeShapeGamma<T> {
    /// Constructs a gamma distribution with the specified shape and scale.
    fn new(shape: T, scale: T) -> Result<Self, GammaError> {
        if shape < T::ONE {
            return Err(GammaError::BadShape);
        }
        if scale <= T::ZERO {
            return Err(GammaError::BadScale);
        }

        // The left/right tail positions are determined by keeping the tail
        // sampling probability roughly constant, using the Wilson and Hilferty
        // approximation.
        let normal_variance = T::ONE / (T::from(9.0) * shape);
        let normal_mean = T::ONE - normal_variance;
        let normal_tail_pos_delta = T::NORMALIZED_TAIL_POS * normal_variance.sqrt();
        let normal_right_tail_pos = normal_mean + normal_tail_pos_delta;
        let right_tail_pos =
            scale * shape * (normal_right_tail_pos * normal_right_tail_pos * normal_right_tail_pos);
        let normal_left_tail_pos = normal_mean - normal_tail_pos_delta;
        let left_tail_pos = scale
            * shape
            * (normal_left_tail_pos * normal_left_tail_pos * normal_left_tail_pos);
    
        // For moderate values of the shape parameter, only the right tail is necessary.
        let (left_tail_pos, tail, tail_area) = if left_tail_pos <= T::ZERO {
            let (tail_func, tail_area) =
                LargeShapeSingleTail::new_with_area(shape, scale, right_tail_pos);
            (T::ZERO, LargeShapeTail::Single(tail_func), tail_area)
        } else {
            let (tail_func, tail_area) =
                LargeShapeDoubleTail::new_with_area(shape, scale, left_tail_pos, right_tail_pos);
            (left_tail_pos, LargeShapeTail::Double(tail_func), tail_area)
        };
        let pdf = LargeShapeUnscaledPdf::new(shape, scale);
        let dpdf = pdf.derivative();
        let init_nodes = util::midpoint_prepartition(&pdf, left_tail_pos, right_tail_pos, 0);
        let extrema: &[T] = &[scale * (shape - T::ONE)];
        let table =
            util::newton_tabulation(&pdf, &dpdf, &init_nodes, extrema, T::TOLERANCE, T::ONE, 50)
                .map_err(|_| GammaError::TabulationFailure)?;

        Ok(Self {
            inner: DistAnyTailed::new(pdf, &table, tail, tail_area),
        })
    }
}
impl<T: GammaFloat> Distribution<T> for LargeShapeGamma<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}

/// Non-normalized gamma probability distribution function.
///
/// ```text
/// f(x) = a x^(k - 1) exp(-x / θ)
///      = exp(-x / θ + (k - 1) ln(x) + ln(a))
/// ```
///
/// with:
///
/// ```text
/// ln(a) = (k - 1)(1 - ln(θ (k - 1)))
/// ```
#[derive(Copy, Clone, Debug)]
struct LargeShapeUnscaledPdf<T> {
    m: T,         // shape - 1
    inv_scale: T, // 1 / scale
    ln_a: T,      // (shape - 1)(1 - ln(scale (shape - 1)))
}
impl<T: Float> LargeShapeUnscaledPdf<T> {
    fn new(shape: T, scale: T) -> Self {
        let m = shape - T::ONE;
        let ln_a = if m > T::ZERO {
            m * (T::ONE - (scale * m).ln())
        } else {
            T::ZERO
        };
        Self {
            m,
            inv_scale: T::ONE / scale,
            ln_a,
        }
    }
    fn derivative(self) -> impl Fn(T) -> T {
        move |x| {
            (self.m - self.inv_scale * x)
                * (x.ln() * (self.m - T::ONE) - x * self.inv_scale + self.ln_a).exp()
        }
    }
}
impl<T: Float> UnivariateFn<T> for LargeShapeUnscaledPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        if x == T::ZERO {
            T::ZERO
        } else {
            (x.ln() * self.m - x * self.inv_scale + self.ln_a).exp()
        }
    }
}

/// Tail envelope for the gamma distribution.
///
/// The envelope function is of the form:
///
/// ```text
/// f(x) = w exp(-(x / x₀) / b)
/// ```
///
/// where `x₀` is the tail cut-in position and where `w` and `b` are chosen such
/// that `f` and its derivative match the gamma probability distribution
/// function at the cut-in location.
#[derive(Copy, Clone, Debug)]
struct LargeShapeSingleTail<T> {
    cut_in: T,
    m: T,
    b: T,
}
impl<T: Float> LargeShapeSingleTail<T> {
    fn new_with_area(shape: T, scale: T, cut_in: T) -> (Self, T) {
        let m = shape - T::ONE;
        let c = cut_in / scale;
        let b = T::ONE / (c - m);
        let w = if m > T::ZERO {
            (m * (c / m).ln()).exp()
        } else {
            T::ONE
        };

        let tail = Self { cut_in, m, b };

        let area = cut_in * w * b.abs() * (m - c).exp(); // works for both left and right tails.

        (tail, area)
    }
}
impl<T: Float> Envelope<T> for LargeShapeSingleTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        let rel_x = T::ONE - self.b * (T::ONE - T::gen(rng)).ln();

        // It is necessary to discard negative `x` values (which may happen with
        // a left tail) before evaluating `ln(x)`.
        if rel_x <= T::ZERO {
            return None;
        }
        let p = (self.m + self.m * (rel_x.ln() - rel_x)).exp();
        if p > T::gen(rng) {
            Some(rel_x * self.cut_in)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct LargeShapeDoubleTail<T> {
    left: LargeShapeSingleTail<T>,
    right: LargeShapeSingleTail<T>,
    left_tail_weight: T,
}
impl<T: Float> LargeShapeDoubleTail<T> {
    fn new_with_area(shape: T, scale: T, left_cut_in: T, right_cut_in: T) -> (Self, T) {
        let (left, left_area) = LargeShapeSingleTail::new_with_area(shape, scale, left_cut_in);
        let (right, right_area) = LargeShapeSingleTail::new_with_area(shape, scale, right_cut_in);

        let area = left_area + right_area;
        let left_tail_weight = left_area / area;

        let tail = Self {
            left,
            right,
            left_tail_weight,
        };

        (tail, area)
    }
}
impl<T: Float> Envelope<T> for LargeShapeDoubleTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        if T::gen(rng) < self.left_tail_weight {
            self.left.try_sample(rng)
        } else {
            self.right.try_sample(rng)
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum LargeShapeTail<T> {
    Single(LargeShapeSingleTail<T>),
    Double(LargeShapeDoubleTail<T>),
}
impl<T: Float> Envelope<T> for LargeShapeTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        match self {
            Self::Single(tail) => tail.try_sample(rng),
            Self::Double(tail) => tail.try_sample(rng),
        }
    }
}
