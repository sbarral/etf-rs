use crate::num::Float;
use crate::primitives::partition::*;
use crate::primitives::*;

use rand_core::RngCore;
use thiserror::Error;

/// A floating point type for use with normal distributions.
pub trait NormalFloat: Float {
    #[doc(hidden)]
    type P: Partition<Self> + ValidSymmetricPartitionSize<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const TAIL_POS: Self;
}

impl NormalFloat for f32 {
    #[doc(hidden)]
    type P = P128<f32>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-4;
    #[doc(hidden)]
    const TAIL_POS: Self = 3.25;
}

impl NormalFloat for f64 {
    #[doc(hidden)]
    type P = P256<f64>;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const TAIL_POS: Self = 3.25;
}

/// Error type for normal distribution construction failures.
#[derive(Error, Debug)]
pub enum NormalError {
    /// The ETF table could not be computed for the provided distribution parameters.
    #[error("could not compute an ETF table for the provided distribution parameters")]
    TabulationFailure,
    /// The provided standard deviation is not strictly positive.
    #[error("the standard deviation should be strictly positive")]
    BadStdDev,
}

/// Normal distribution with arbitrary mean and standard deviation.
#[derive(Clone)]
pub struct Normal<T: NormalFloat> {
    inner: DistSymmetricTailed<T::P, T, UnscaledNormalPdf<T>, NormalTailEnvelope<T>>,
}

impl<T: NormalFloat> Normal<T> {
    /// Constructs a normal distribution with the specified mean and standard deviation.
    pub fn new(mean: T, std_dev: T) -> Result<Self, NormalError> {
        if std_dev <= T::ZERO {
            return Err(NormalError::BadStdDev);
        }
        let two_alpha = -T::ONE / (std_dev * std_dev);
        let alpha = T::from(0.5_f32) * two_alpha;
        let pdf = UnscaledNormalPdf { mean, alpha };
        let dpdf = move |x: T| {
            let dx = x - mean;
            dx * two_alpha * (dx * dx * alpha).exp()
        };
        let (table, tail_func, tail_area) = normal_parts(mean, std_dev, pdf, dpdf)?;

        Ok(Self {
            inner: DistSymmetricTailed::new(mean, pdf, &table, tail_func, tail_area),
        })
    }
}

impl<T: NormalFloat> Distribution<T> for Normal<T> {
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}

/// Central normal distribution with arbitrary standard deviation.
#[derive(Clone)]
pub struct CentralNormal<T: NormalFloat> {
    inner: DistCentralTailed<T::P, T, UnscaledCentralNormalPdf<T>, NormalTailEnvelope<T>>,
}

impl<T: NormalFloat> CentralNormal<T> {
    pub fn new(std_dev: T) -> Result<Self, NormalError> {
        if std_dev <= T::ZERO {
            return Err(NormalError::BadStdDev);
        }
        let two_alpha = -T::ONE / (std_dev * std_dev);
        let alpha = T::from(0.5_f32) * two_alpha;
        let pdf = UnscaledCentralNormalPdf { alpha };
        let dpdf = move |x: T| x * two_alpha * (x * x * alpha).exp();
        let (table, tail_func, tail_area) = normal_parts(T::ZERO, std_dev, pdf, dpdf)?;
        Ok(Self {
            inner: DistCentralTailed::new(pdf, &table, tail_func, tail_area),
        })
    }
}

impl<T: NormalFloat> Distribution<T> for CentralNormal<T> {
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}

/// Non-normalized normal probability distribution function with arbitrary mean
/// and standard deviation.
#[derive(Copy, Clone, Debug)]
struct UnscaledNormalPdf<T> {
    mean: T,
    alpha: T, // -1/(2*std_dev^2)
}

impl<T: Float> UnivariateFn<T> for UnscaledNormalPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        let dx = x - self.mean;

        (self.alpha * dx * dx).exp()
    }
}

#[derive(Copy, Clone, Debug)]
struct UnscaledCentralNormalPdf<T> {
    alpha: T,
}

impl<T: Float> UnivariateFn<T> for UnscaledCentralNormalPdf<T> {
    #[inline]
    fn eval(&self, x: T) -> T {
        (self.alpha * x * x).exp()
    }
}

#[derive(Copy, Clone, Debug)]
struct NormalTailEnvelope<T> {
    cut_in: T,
    a_x: T,
    a_y: T,
}

impl<T: Float> NormalTailEnvelope<T> {
    fn new(mean: T, std_dev: T, cut_in: T) -> Self {
        Self {
            cut_in,
            a_x: std_dev * std_dev / (cut_in - mean),
            a_y: T::from(-2_f32) * std_dev * std_dev,
        }
    }
}

impl<T: Float> Envelope<T> for NormalTailEnvelope<T> {
    #[inline]
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        loop {
            let x = (T::ONE - T::gen(rng)).ln() * self.a_x;
            let y = (T::ONE - T::gen(rng)).ln() * self.a_y;
            if y >= x * x {
                return Some(self.cut_in - x);
            }
        }
    }
}

fn normal_parts<T: NormalFloat, F: UnivariateFn<T>, DF: UnivariateFn<T>>(
    mean: T,
    std_dev: T,
    pdf: F,
    dpdf: DF,
) -> Result<(InitTable<T::P, T>, NormalTailEnvelope<T>, T), NormalError> {
    let tail_position = mean + T::TAIL_POS * std_dev;

    let inv_sqrt_two = T::from(0.5f32).sqrt();
    let tail_area = T::PI.sqrt() * std_dev * inv_sqrt_two * (T::TAIL_POS * inv_sqrt_two).erfc();

    // Build the distribution.
    let init_nodes = util::midpoint_prepartition(&pdf, mean, tail_position, 0);
    let table = util::newton_tabulation(&pdf, &dpdf, &init_nodes, &[], T::TOLERANCE, T::ONE, 10)
        .map_err(|_| NormalError::TabulationFailure)?;
    let tail_func = NormalTailEnvelope::new(mean, std_dev, tail_position);

    Ok((table, tail_func, tail_area))
}
