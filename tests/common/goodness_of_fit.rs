use super::util::{test_rng, TestFloat};
use etf::num::Float;
use etf::primitives::Distribution;

/// A set of sampling bins regularly distributed between `x0` and `x1`.
///
/// Samples which do not fall within interval `[x0, x1]` are accumulated into
/// the residual.
pub struct Histogram {
    x0: f64,
    x1: f64,
    scale: f64,
    bin_count: f64,
    bins: Vec<u64>,
    residual: u64,
}

#[allow(dead_code)]
impl Histogram {
    pub fn new(x0: f64, x1: f64, bin_count: usize) -> Self {
        if bin_count < 1 {
            panic!("Histogram must contain at least one bin");
        }
        let mut bins = Vec::with_capacity(bin_count);
        bins.resize(bin_count, 0);
        let bin_count = bin_count as f64;
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
    pub fn add(&mut self, x: f64) {
        let i = (x - self.x0) * self.scale;
        if i >= 0.0 && i < self.bin_count {
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
    pub fn x0(&self) -> f64 {
        self.x0
    }
    pub fn x1(&self) -> f64 {
        self.x1
    }
}

/// Returns the upper tail P-value of a chi-square test.
///
/// The number of degrees of freedom `k` is assumed sufficiently large to
/// approximate the chi-square distribution with a normal distribution when
/// computing the p-value.
#[allow(dead_code)]
pub fn chi_square_test<F: Fn(f64) -> f64>(histogram: Histogram, cdf: F) -> f64 {
    let bins = histogram.bins();
    let x0 = histogram.x0();
    let x1 = histogram.x1();
    let n = bins.iter().fold(0, |sum, count| sum + count) + histogram.residual(); // sample count
    let m = bins.len(); // bin count
    let mut k = m - 1; // degrees of freedom
    let n = n as f64;

    // Compute χ².
    let mut chi_square = 0.0;
    // Contribution to χ² over interval [x0, x1].
    let mut cdf_l = cdf(x0);
    for i in 0..m {
        let x = x1 - (m - i - 1) as f64 / m as f64 * (x1 - x0);
        let cdf_r = cdf(x);
        let expected = (cdf_r - cdf_l) * n;
        cdf_l = cdf_r;
        let delta = bins[i] as f64 - expected;
        chi_square = chi_square + delta * delta / expected;
    }
    // Account for the contribution of the residual to χ² if the expected
    // residual is at least equal to 1 sample.
    let expected_residual = (cdf(x0) + 1.0 - cdf(x1)) * n;
    if expected_residual > 1.0 {
        let delta = histogram.residual() as f64 - expected_residual;
        chi_square = chi_square + delta * delta / expected_residual;
        k = k + 1; // increase degrees of freedom
    }

    // Assume that `k` is large enough to approximate the χ² distribution with a
    // normal distribution.
    let k = k as f64;
    let p_value = (1.0 - ((chi_square - k) / (2.0 * k.sqrt())).erf()) / 2.0;

    p_value
}

/// Assess goodness of fit based on a χ² test.
#[allow(dead_code)]
pub fn goodness_of_fit<T: TestFloat, D: Distribution<T>, F: Fn(f64) -> f64>(
    distribution: D,
    cdf: F,
    x0: f64,
    x1: f64,
    sample_count: u64,
    bin_count: usize,
    p_value_threshold: f64,
) {
    // Sample the distribution.
    let mut histogram = Histogram::new(x0, x1, bin_count);
    let mut rng = test_rng();
    
    for _ in 0..sample_count {
        let r = distribution.sample(&mut rng);
        histogram.add(r.as_f64());
    }

    // Process the data.
    let p_value = chi_square_test(histogram, cdf);
    println!("P-value: {}", p_value);

    assert!(p_value > p_value_threshold);
}
