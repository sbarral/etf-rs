use crate::num::Float;
use crate::primitives::*;
use rand_core::RngCore;

/// Non-normalized normal probability distribution function with arbitrary mean
/// and standard deviation.
#[derive(Copy, Clone, Debug)]
struct NormalPdf<T> {
    mean: T,
    alpha: T,
}

impl<T: Float> Func<T> for NormalPdf<T> {
    #[inline(always)]
    fn eval(&self, x: T) -> T {
        let dx = x - self.mean;

        (self.alpha * dx * dx).exp()
    }
}

/// Non-normalized central normal probability distribution function with
/// arbitrary standard deviation.
#[derive(Copy, Clone, Debug)]
struct CentralNormalPdf<T> {
    alpha: T,
}

impl<T: Float> Func<T> for CentralNormalPdf<T> {
    #[inline(always)]
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
    #[inline(always)]
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

pub trait NormalFloat: Float {
    #[doc(hidden)]
    type P: partition::Partition + ValidSymmetricPartitionSize<Self>;
    #[doc(hidden)]
    const TOLERANCE: Self;
    #[doc(hidden)]
    const TAIL_POS: Self;
}

impl NormalFloat for f32 {
    #[doc(hidden)]
    type P = partition::P128;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-4;
    #[doc(hidden)]
    const TAIL_POS: Self = 3.25;
}

impl NormalFloat for f64 {
    #[doc(hidden)]
    type P = partition::P256;
    #[doc(hidden)]
    const TOLERANCE: Self = 1.0e-6;
    #[doc(hidden)]
    const TAIL_POS: Self = 3.25;
}

fn new_normal<T: Float + NormalFloat, F: Func<T>, DF: Func<T>>(
    mean: T,
    std_dev: T,
    pdf: F,
    dpdf: DF,
) -> (
    partition::InitTable<T::P, T>,
    NormalTailEnvelope<T>,
    T,
) {
    let tail_position = mean + T::TAIL_POS * std_dev;

    let one_half_sqrt_pi = T::from(0.5f32) * T::PI.sqrt();
    let sigma_sqrt_two = std_dev * T::from(2f32).sqrt();
    let tail_area =
        one_half_sqrt_pi * sigma_sqrt_two * ((tail_position - mean) / sigma_sqrt_two).erfc();

    // Build the distribution.
    let init_nodes = util::midpoint_prepartition(&pdf, mean, tail_position, 0);
    let table =
        util::newton_tabulation(&pdf, &dpdf, &init_nodes, &[], T::TOLERANCE, T::ONE, 10).unwrap();
    let tail_func = NormalTailEnvelope::new(mean, std_dev, tail_position);

    (table, tail_func, tail_area)
}

/// Normal distribution with arbitrary mean and standard deviation.
pub struct Normal<T: Float + NormalFloat> {
    inner: DistSymmetricTailed<T::P, T, NormalPdf<T>, NormalTailEnvelope<T>>,
}

impl<T: Float + NormalFloat> Normal<T> {
    pub fn new(mean: T, std_dev: T) -> Self {
        let two_alpha = -T::ONE / (std_dev * std_dev);
        let alpha = T::from(0.5_f32) * two_alpha;
        let pdf = NormalPdf { mean, alpha };
        let dpdf = move |x: T| {
            let dx = x - mean;
            dx * two_alpha * (dx * dx * alpha).exp()
        };
    
        let (table, tail_func, tail_area) = new_normal(mean, std_dev, pdf, dpdf);

        Self {
            inner: DistSymmetricTailed::new(mean, pdf, &table, tail_func, tail_area),
        }
    }
}

impl<T: Float + NormalFloat> Distribution<T> for Normal<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}

/// Central normal distribution with arbitrary standard deviation.
pub struct CentralNormal<T: Float + NormalFloat> {
    inner: DistCentralTailed<T::P, T, CentralNormalPdf<T>, NormalTailEnvelope<T>>,
}

impl<T: Float + NormalFloat> CentralNormal<T> {
    pub fn new(std_dev: T) -> Self {
        let two_alpha = -T::ONE / (std_dev * std_dev);
        let alpha = T::from(0.5_f32) * two_alpha;
        let pdf = CentralNormalPdf { alpha };
        let dpdf = move |x: T| {
            x * two_alpha * (x * x * alpha).exp()
        };
    
        let (table, tail_func, tail_area) = new_normal(T::ZERO, std_dev, pdf, dpdf);
        
        Self {
            inner: DistCentralTailed::new(pdf, &table, tail_func, tail_area),
        }
    }
}

impl<T: Float + NormalFloat> Distribution<T> for CentralNormal<T> {
    #[inline(always)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        self.inner.sample(rng)
    }
}
