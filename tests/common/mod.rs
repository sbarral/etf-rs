use libm;
use std::fmt::Display;

use etf::Float;
use rand::distributions::Distribution;
use rand_pcg;

pub trait TestFloat: Float + Display {
    fn erf(self) -> Self;
    fn cast_u64(u: u64) -> Self;
}
impl TestFloat for f32 {
    fn erf(self) -> Self {
        libm::erff(self)
    }
    fn cast_u64(u: u64) -> Self {
        u as f32
    }
}
impl TestFloat for f64 {
    fn erf(self) -> Self {
        libm::erf(self)
    }
    fn cast_u64(u: u64) -> Self {
        u as f64
    }
}

/// A set of sampling bins regularly distributed between `x0` and `x1`.
///
/// Samples which do not fall within interval `[x0, x1]` are accumulated into
/// the residual.
pub struct Histogram<T> {
    x0: T,
    x1: T,
    scale: T,
    bin_count: T,
    bins: Vec<u64>,
    residual: u64,
}

#[allow(dead_code)]
impl<T: Float> Histogram<T> {
    pub fn new(x0: T, x1: T, bin_count: usize) -> Self {
        if bin_count < 1 {
            panic!("Histogram must contain at least one bin");
        }
        let mut bins = Vec::with_capacity(bin_count);
        bins.resize(bin_count, 0);
        let bin_count = T::cast_usize(bin_count);
        let scale = bin_count / (x1 - x0);
        Self {
            x0,
            x1,
            scale,
            bin_count,
            bins,
            residual: 0,
        }
    }
    pub fn add(&mut self, x: T) {
        let i = (x - self.x0) * self.scale;
        if i >= T::zero() && i < self.bin_count {
            let i = i.as_usize();
            self.bins[i] = self.bins[i] + 1;
        } else {
            self.residual = self.residual + 1;
        }
    }
    pub fn bins(&self) -> &[u64] {
        &self.bins
    }
    pub fn residual(&self) -> u64 {
        self.residual
    }
    pub fn x0(&self) -> T {
        self.x0
    }
    pub fn x1(&self) -> T {
        self.x1
    }
}

/// Returns the upper tail P-value of a chi-square test.
///
/// The number of degrees of freedom `k` is assumed sufficiently large to
/// approximate the chi-square distribution with a normal distribution when
/// computing the p-value.
pub fn chi_square_test<F: Fn(T) -> T, T: TestFloat>(histogram: Histogram<T>, cdf: F) -> T {
    let bins = histogram.bins();
    let x0 = histogram.x0();
    let x1 = histogram.x1();
    let n = bins.iter().fold(0, |sum, count| sum + count) + histogram.residual(); // sample count
    let m = bins.len(); // bin count
    let mut k = m - 1; // degrees of freedom
    let n = T::cast_u64(n);

    // Compute χ².
    let mut chi_square = T::zero();
    // Contribution to χ² over interval [x0, x1].
    let mut cdf_l = cdf(x0);
    for i in 0..m {
        let x = x1 - T::cast_usize(m - i - 1) / T::cast_usize(m) * (x1 - x0);
        let cdf_r = cdf(x);
        let expected = (cdf_r - cdf_l) * n;
        cdf_l = cdf_r;
        let delta = T::cast_u64(bins[i]) - expected;
        chi_square = chi_square + delta * delta / expected;
    }
    // Account for the contribution of the residual to χ² if the expected
    // residual is at least equal to 1 sample.
    let expected_residual = (cdf(x0) + T::ONE - cdf(x1)) * n;
    if expected_residual > T::ONE {
        let delta = T::cast_u64(histogram.residual()) - expected_residual;
        chi_square = chi_square + delta * delta / expected_residual;
        k = k + 1; // increase degrees of freedom
    }

    // Assume that `k` is large enough to approximate the χ² distribution with a
    // normal distribution.
    let two = T::ONE + T::ONE;
    let k = T::cast_usize(k);
    let p_value = (T::ONE - ((chi_square - k) / (two * k.sqrt())).erf()) / two;

    p_value
}

/// Assess goodness of fit based on a χ² test.
pub fn goodness_of_fit<T: TestFloat, D: Distribution<T>, F: Fn(T) -> T>(
    distribution: D,
    cdf: F,
    x0: T,
    x1: T,
    sample_count: u64,
    bin_count: usize,
    p_value_threshold: T,
) {
    // Sample the distribution.
    let mut histogram = Histogram::new(x0, x1, bin_count);
    // Use PCG64 with the author's recommended seed.
    let mut rng = rand_pcg::Lcg128Xsl64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    //let mut rng = rand::thread_rng();
    for _ in 0..sample_count {
        let r = distribution.sample(&mut rng);
        histogram.add(r);
    }

    // Process the data.
    let p_value = chi_square_test(histogram, cdf);

    assert!(p_value > p_value_threshold);
}
