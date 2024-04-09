use super::{GammaError, GammaFloat};
use crate::num::Float;
use crate::primitives::*;

use rand_core::RngCore;

/// Specialized gamma distribution for `k≥1`.
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
pub struct LargeShapeGamma<T: GammaFloat> {
    inner: DistAnyTailed<T::P, T, LargeShapeUnscaledPdf<T>, LargeShapeTail<T>>,
}
impl<T: GammaFloat> LargeShapeGamma<T> {
    /// Constructs a gamma distribution with the specified shape and scale.
    pub fn new(shape: T, scale: T) -> Result<Self, GammaError> {
        // The left/right tail positions are determined by keeping the tail
        // sampling probability roughly constant, using the Wilson and Hilferty
        // approximation.
        let normal_variance = T::ONE / (T::from(9.0) * shape);
        let normal_mean = T::ONE - normal_variance;
        let normal_tail_pos_delta = T::LARGE_SHAPE_NORMALIZED_TAIL_POS * normal_variance.sqrt();
        let normal_right_tail_pos = normal_mean + normal_tail_pos_delta;
        let right_tail_pos =
            scale * shape * (normal_right_tail_pos * normal_right_tail_pos * normal_right_tail_pos);
        let normal_left_tail_pos = normal_mean - normal_tail_pos_delta;
        let left_tail_pos =
            scale * shape * (normal_left_tail_pos * normal_left_tail_pos * normal_left_tail_pos);
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

/// Non-normalized gamma probability distribution function for `k≥1`.
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
///
/// In order to improve floating-point stability at large `k` without
/// introducing a singularity at `k=2`, `f` is rewritten in terms of
/// variable `xs`:
///
/// ```text
/// xs = x / (m θ) with m = max(k-1, 1)
/// ```
///
/// in the form:
///
/// ```text
/// f(x) = exp(m ((δ - xs) + β ln(xs)))
/// ```
///
#[derive(Copy, Clone, Debug)]
struct LargeShapeUnscaledPdf<T> {
    m: T,       // max(shape - 1, 1)
    scaling: T, // 1 / (scale * m)
    beta: T,    // (shape - 1) / m
    delta: T,   // beta * (1 - ln(beta))
}
impl<T: Float> LargeShapeUnscaledPdf<T> {
    fn new(shape: T, scale: T) -> Self {
        let m = (shape - T::ONE).max(T::ONE);
        let beta = (shape - T::ONE) / m;
        Self {
            m,
            scaling: T::ONE / (scale * m),
            beta: (shape - T::ONE) / m,
            delta: if beta <= T::ZERO {
                T::ZERO
            } else {
                beta * (T::ONE - beta.ln())
            },
        }
    }
    fn derivative(self) -> impl Fn(T) -> T {
        move |x| {
            let xs = x * self.scaling;
            let ln_xs = xs.ln();

            self.m
                * self.scaling
                * (self.beta - xs)
                * (self.m * ((self.delta - xs) + self.beta * ln_xs) - ln_xs).exp()
        }
    }
}
impl<T: Float> UnivariateFn<T> for LargeShapeUnscaledPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        let xs = x * self.scaling;

        (self.m * ((self.delta - xs) + self.beta * xs.ln())).exp()
    }
}

/// Tail envelope of the gamma distribution for `k≥1`.
#[derive(Copy, Clone, Debug)]
enum LargeShapeTail<T> {
    Single(LargeShapeSingleTail<T>),
    Double(LargeShapeDoubleTail<T>),
}
impl<T: Float> TryDistribution<T> for LargeShapeTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        match self {
            Self::Single(tail) => tail.try_sample(rng),
            Self::Double(tail) => tail.try_sample(rng),
        }
    }
}

/// Left or right tail envelope of the gamma distribution for `k≥1`.
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

        let tail = Self { cut_in, m, b };

        let m_ln_m = if m <= T::ZERO { T::ZERO } else { m * m.ln() };
        let area = scale * (shape * c.ln() - m_ln_m - (c - m)).exp() / (c - m).abs();

        (tail, area)
    }
}
impl<T: Float> TryDistribution<T> for LargeShapeSingleTail<T> {
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

/// Combined left & right tail envelope of the gamma distribution for `k≥1`.
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
impl<T: Float> TryDistribution<T> for LargeShapeDoubleTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        if T::gen(rng) < self.left_tail_weight {
            self.left.try_sample(rng)
        } else {
            self.right.try_sample(rng)
        }
    }
}
