use crate::num::Float;
use crate::primitives::*;
use rand_core::RngCore;

/// Non-normalized Cauchy probability distribution function with arbitrary
/// location and scale.
#[derive(Copy, Clone, Debug)]
struct UnscaledCauchyPdf<T> {
    location: T,
    square_inv_scale: T,
}

impl<T: Float> UnivariateFn<T> for UnscaledCauchyPdf<T> {
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
struct CauchyTailEnvelope<T> {
    location: T,
    scale: T,
    a: T,
    b: T,
}

impl<T: Float> CauchyTailEnvelope<T> {
    fn new(location: T, scale: T, cut_in: T) -> Self {
        let fmin = T::atan((cut_in - location) / scale) / T::PI + T::from(0.5);

        Self {
            location,
            scale,
            a: T::PI * (T::ONE - fmin),
            b: T::PI * (fmin - T::from(0.5)),
        }
    }
}

impl<T: Float> Envelope<T> for CauchyTailEnvelope<T> {
    #[inline(always)]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        Some(self.location + self.scale * T::tan(self.a * T::gen(rng) + self.b))
    }
}

pub trait CauchyFloat: Float {
    #[doc(hidden)]
    type P: partition::Partition<Self> + ValidSymmetricPartitionSize<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const TAIL_POS: Self;
}

impl CauchyFloat for f32 {
    #[doc(hidden)]
    type P = partition::P128<f32>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-4;
    #[doc(hidden)]
    const TAIL_POS: Self = 3.25;
}

impl CauchyFloat for f64 {
    #[doc(hidden)]
    type P = partition::P256<f64>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const TAIL_POS: Self = 400.0;
}

/// Normal distribution with arbitrary mean and arbitrary standard deviation.
pub struct Cauchy<T: CauchyFloat> {
    inner: DistSymmetricTailed<T::P, T, UnscaledCauchyPdf<T>, CauchyTailEnvelope<T>>,
}

impl<T: CauchyFloat> Cauchy<T> {
    pub fn new(location: T, scale: T) -> Self {
        let square_inv_scale = T::ONE / (scale * scale);
        let pdf = UnscaledCauchyPdf {
            location,
            square_inv_scale,
        };
        let dpdf = move |x: T| {
            let dx = x - location;

            let minus_dv = T::from(-2.0) * square_inv_scale * dx;
            let v = T::ONE + square_inv_scale * dx * dx;

            minus_dv / (v * v)
        };

        let tail_position = location + T::TAIL_POS * scale;
        let tail_area = scale * (T::atan(-T::TAIL_POS) + T::from(0.5) * T::PI);
        let init_nodes = util::midpoint_prepartition::<T::P, _, _>(&pdf, location, tail_position, 0);
        let table =
            util::newton_tabulation(&pdf, &dpdf, &init_nodes, &[], T::TOLERANCE, T::ONE, 10)
                .unwrap();
        let tail_func = CauchyTailEnvelope::new(location, scale, tail_position);
        Self {
            inner: DistSymmetricTailed::new(location, pdf, &table, tail_func, tail_area),
        }
    }
}

impl<T: CauchyFloat> Distribution<T> for Cauchy<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}
