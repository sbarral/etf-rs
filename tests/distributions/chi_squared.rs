use crate::common::{collisions, fair_goodness_of_fit, TestFloat};
use etf::distributions::{ChiSquared, ChiSquaredFloat};

#[cfg(all(feature = "rand_distribution"))]
use rand_distr;

fn chi_squared_cdf(x: f64, k: f64) -> f64 {
    use special::Gamma;
    (0.5 * x).inc_gamma(0.5 * k)
}

fn chi_squared_collisions<T: TestFloat + ChiSquaredFloat>(k: T) {
    collisions(
        ChiSquared::new(k).unwrap(),
        |x| chi_squared_cdf(x, k.into()),
        20,
        64,
        10,
        0.05,
    );
}

fn chi_squared_fit<T: TestFloat + ChiSquaredFloat>(k: T) {
    fair_goodness_of_fit(
        ChiSquared::new(k).unwrap(),
        |x| chi_squared_cdf(x, k.into()),
        50_000_000,
        401,
        0.01,
    );
}

macro_rules! test_case {
    ($ffit:ident, $fcoll:ident, $v:expr) => {
        #[test]
        fn $ffit() {
            chi_squared_fit($v);
        }
        #[test]
        fn $fcoll() {
            chi_squared_collisions($v);
        }
    };
}
macro_rules! rand_test_case {
    ($ffit:ident, $fcoll:ident, $v:expr) => {
        #[cfg(feature = "rand_distribution")]
        #[test]
        fn $ffit() {
            fair_goodness_of_fit(
                rand_distr::ChiSquared::new($v).unwrap(),
                |x| chi_squared_cdf(x, $v as f64),
                50_000_000,
                401,
                0.01,
            );
        }
        #[cfg(feature = "rand_distribution")]
        #[test]
        fn $fcoll() {
            collisions(
                rand_distr::ChiSquared::new($v).unwrap(),
                |x| chi_squared_cdf(x, $v as f64),
                20,
                64,
                10,
                0.05,
            );
        }
    };
}

// Tests for very small k fail even though the distributions are apparently OK;
// this seems to be due to floating-point precision issues that arise during
// testing, in particular in the evaluation of the lower regularized incomplete
// gamma function.

test_case!(
    chi_squared_64_fit_k0_02,
    chi_squared_64_collisions_k0_02,
    0.02_f64
); // fails for f32, but so does rand_distr::ChiSquared
test_case!(
    chi_squared_32_fit_k0_5,
    chi_squared_32_collisions_k0_5,
    0.5_f32
);
test_case!(
    chi_squared_64_fit_k0_5,
    chi_squared_64_collisions_k0_5,
    0.5_f64
);
test_case!(chi_squared_32_fit_k2, chi_squared_32_collisions_k2, 2_f32);
test_case!(chi_squared_64_fit_k2, chi_squared_64_collisions_k2, 2_f64);
test_case!(
    chi_squared_32_fit_k4_5,
    chi_squared_32_collisions_k4_5,
    4.5_f32
);
test_case!(
    chi_squared_64_fit_k4_5,
    chi_squared_64_collisions_k4_5,
    4.5_f64
);
test_case!(
    chi_squared_32_fit_k10000,
    chi_squared_32_collisions_k10000,
    10_000_f32
);
test_case!(
    chi_squared_64_fit_k10000,
    chi_squared_64_collisions_k10000,
    10_000_f64
); // fails for f32

rand_test_case!(
    rand_chi_squared_32_fit_k0_02,
    rand_chi_squared_32_collisions_k0_02,
    0.02_f32
);
rand_test_case!(
    rand_chi_squared_64_fit_k0_02,
    rand_chi_squared_64_collisions_k0_02,
    0.02_f64
);
rand_test_case!(
    rand_chi_squared_32_fit_k10000,
    rand_chi_squared_32_collisions_k10000,
    10_000_f32
);
rand_test_case!(
    rand_chi_squared_64_fit_k10000,
    rand_chi_squared_64_collisions_k10000,
    10_000_f64
);
