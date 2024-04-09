use super::{GammaError, GammaFloat};
use crate::num::Float;
use crate::primitives::*;

use rand_core::RngCore;

/// Specialized gamma distribution for `k<1`.
///
/// For `k<1`, the implementation introduces variable `X` such that:
///
/// ```text
/// x = exp(X)
/// ```
///
/// which leads to the following non-normalized probability density function:
///
/// ```text
/// f(X) = exp(k X - exp(X) / θ)
/// ```
///
/// The left tail is generated using the envelope function:
///
/// ```text
/// fl(X) = exp(k X)  with X ≤ Xl
/// ```
///
/// The position of the left tail is specified through the relative weight `Wl`
/// of the left tail envelope and is relatively well approximated by:
///
/// ```text
/// Xl ≈ ln(θ) + ln(Wl) / k
/// ```
///
/// The right tail is in turn generated using the envelope function:
///
/// ```text
/// fr(X) = exp(k Xr - exp(X) / θ)   with X ≥ Xr
///
/// The following choice of the right tail position ensures that the relative
/// weight `Wr` of the actual right tail is below the specified threshold:
///
/// ```text
/// Xr = ln(θ max[1, ln(k / (0.8856 Wr))])
/// ```
///
#[derive(Clone)]
pub struct SmallShapeGamma<T: GammaFloat> {
    inner: DistAnyTailed<T::P, T, SmallShapeUnscaledPdf<T>, SmallShapeTail<T>>,
}
impl<T: GammaFloat> SmallShapeGamma<T> {
    /// Constructs a gamma distribution with the specified shape and scale.
    pub fn new(shape: T, scale: T) -> Result<Self, GammaError> {
        let left_tail_pos = scale.ln() + T::SMALL_SHAPE_LEFT_TAIL_ENVELOPE_PROBABILITY.ln() / shape;
        let right_tail_pos = (T::ONE
            .max((shape / (T::from(0.8856) * T::SMALL_SHAPE_RIGHT_TAIL_MAX_PROBABILITY)).ln())
            * scale)
            .ln();

        let (tail, tail_area) =
                SmallShapeTail::new_with_area(shape, scale, left_tail_pos, right_tail_pos);
        let pdf = SmallShapeUnscaledPdf::new(shape, scale);
        let dpdf = pdf.derivative();
        let init_nodes = util::midpoint_prepartition(&pdf, left_tail_pos, right_tail_pos, 0);
        let extrema: &[T] = &[(scale * shape).ln()];
        let table =
            util::newton_tabulation(&pdf, &dpdf, &init_nodes, extrema, T::TOLERANCE, T::ONE, 50)
                .map_err(|_| GammaError::TabulationFailure)?;

        Ok(Self {
            inner: DistAnyTailed::new(pdf, &table, tail, tail_area),
        })
    }
}
impl<T: GammaFloat> Distribution<T> for SmallShapeGamma<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng).exp()
    }
}

/// Non-normalized transformed gamma probability distribution function for
/// `k<1`.
///
/// ```text
/// f(X) = exp(k X - exp(X) / θ)
/// ```
#[derive(Copy, Clone, Debug)]
struct SmallShapeUnscaledPdf<T> {
    shape: T,
    ln_scale: T,
}
impl<T: Float> SmallShapeUnscaledPdf<T> {
    fn new(shape: T, scale: T) -> Self {
        Self {
            shape,
            ln_scale: scale.ln(),
        }
    }
    fn derivative(self) -> impl Fn(T) -> T {
        move |x| {
            let exp_x_star = (x - self.ln_scale).exp();

            (self.shape - exp_x_star) * (self.shape * x - exp_x_star).exp()
        }
    }
}
impl<T: Float> UnivariateFn<T> for SmallShapeUnscaledPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        let exp_x_star = (x - self.ln_scale).exp();

        (self.shape * x - exp_x_star).exp()
    }
}

/// Combined left & right tail envelope of the transformed gamma distribution
/// for `k<1`.
#[derive(Copy, Clone, Debug)]
struct SmallShapeTail<T> {
    left: SmallShapeLeftTail<T>,
    right: SmallShapeRightTail<T>,
    left_tail_weight: T,
}
impl<T: Float> SmallShapeTail<T> {
    fn new_with_area(shape: T, scale: T, left_cut_in: T, right_cut_in: T) -> (Self, T) {
        let (left, left_area) = SmallShapeLeftTail::new_with_area(shape, scale, left_cut_in);
        let (right, right_area) = SmallShapeRightTail::new_with_area(shape, scale, right_cut_in);
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
impl<T: Float> TryDistribution<T> for SmallShapeTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        if T::gen(rng) < self.left_tail_weight {
            self.left.try_sample(rng)
        } else {
            self.right.try_sample(rng)
        }
    }
}

/// Left tail envelope of the transformed gamma distribution for `k<1`.
///
/// The envelope function is of the form:
///
/// ```text
/// fl(X) = exp(k X) with X ≤ Xl
/// ```
#[derive(Copy, Clone, Debug)]
struct SmallShapeLeftTail<T> {
    cut_in: T,
    inv_shape: T,
    minus_scale: T,
}
impl<T: Float> SmallShapeLeftTail<T> {
    fn new_with_area(shape: T, scale: T, cut_in: T) -> (Self, T) {
        let tail = Self {
            cut_in,
            inv_shape: T::ONE / shape,
            minus_scale: -scale,
        };
        let area = (shape * cut_in).exp() / shape;

        (tail, area)
    }
}
impl<T: Float> TryDistribution<T> for SmallShapeLeftTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        let x = self.cut_in + self.inv_shape * (T::ONE - T::gen(rng)).ln();

        if x.exp() < self.minus_scale * T::gen(rng).ln() {
            Some(x)
        } else {
            None
        }
    }
}

/// Right tail envelope of the transformed gamma distribution for `k<1`.
///
/// The envelope function is of the form:
///
/// ```text
/// fr(X) = exp(k Xr - exp(X) / θ)   with X ≥ Xr
/// ```
#[derive(Copy, Clone, Debug)]
struct SmallShapeRightTail<T> {
    cut_in: T,
    exp_cut_in: T,
    scale: T,
    m: T, // shape - 1
}
impl<T: Float> SmallShapeRightTail<T> {
    fn new_with_area(shape: T, scale: T, cut_in: T) -> (Self, T) {
        let m = shape - T::ONE;
        let exp_cut_in = cut_in.exp();
        let tail = Self {
            cut_in,
            exp_cut_in,
            scale,
            m,
        };
        let area = scale * (m * cut_in - exp_cut_in / scale).exp();

        (tail, area)
    }
}
impl<T: Float> TryDistribution<T> for SmallShapeRightTail<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        let x = (self.exp_cut_in - self.scale * (T::ONE - T::gen(rng)).ln()).ln();

        if self.m * (x - self.cut_in) > T::gen(rng).ln() {
            Some(x)
        } else {
            None
        }
    }
}
