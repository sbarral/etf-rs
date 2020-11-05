use crate::common::{collisions, goodness_of_fit};
use etf::distributions::{Normal, CentralNormal};
use etf::num::Float;

// CDF for normal distribution.
pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    0.5 * (1.0 + ((0.5_f64).sqrt() * (x - mean) / std_dev).erf())
}

#[test]
fn normal_32_collisions() {
    let mean = -1.7_f64;
    let std_dev = 2.8_f64;

    collisions(
        Normal::new(mean as f32, std_dev as f32).unwrap(),
        |x| normal_cdf(x, mean, std_dev),
        20,
        64,
        10,
        0.05,
    );
}


#[test]
fn normal_64_collisions() {
    let mean = -1.7_f64;
    let std_dev = 2.8_f64;

    collisions(
        Normal::new(mean as f64, std_dev as f64).unwrap(),
        |x| normal_cdf(x, mean, std_dev),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn central_normal_32_collisions() {
    let std_dev = 0.7_f64;

    collisions(
        CentralNormal::new(std_dev as f32).unwrap(),
        |x| normal_cdf(x, 0.0, std_dev),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn central_normal_64_collisions() {
    let std_dev = 0.7_f64;

    collisions(
        CentralNormal::new(std_dev as f64).unwrap(),
        |x| normal_cdf(x, 0.0, std_dev),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn normal_32_fit() {
    let mean = 2.2_f64;
    let std_dev = 3.4_f64;
    let n_sigma = 4.0_f64; // test interval half-width in standard deviation units

    goodness_of_fit(
        Normal::new(mean as f32, std_dev as f32).unwrap(),
        |x| normal_cdf(x, mean, std_dev),
        mean - n_sigma * std_dev,
        mean + n_sigma * std_dev,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn normal_64_fit() {
    let mean = 2.2_f64;
    let std_dev = 3.4_f64;
    let n_sigma = 4.0_f64; // test interval half-width in std. dev. units

    goodness_of_fit(
        Normal::new(mean as f64, std_dev as f64).unwrap(),
        |x| normal_cdf(x, mean, std_dev),
        mean - n_sigma * std_dev,
        mean + n_sigma * std_dev,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn central_normal_32_fit() {
    let std_dev = 1.3_f64;
    let n_sigma = 4.0_f64; // test interval half-width in standard deviation units

    goodness_of_fit(
        CentralNormal::new(std_dev as f32).unwrap(),
        |x| normal_cdf(x, 0.0, std_dev),
        - n_sigma * std_dev,
        n_sigma * std_dev,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn central_normal_64_fit() {
    let std_dev = 1.3_f64;
    let n_sigma = 4.0_f64; // test interval half-width in std. dev. units

    goodness_of_fit(
        CentralNormal::new(std_dev as f64).unwrap(),
        |x| normal_cdf(x, 0.0, std_dev),
        - n_sigma * std_dev,
        n_sigma * std_dev,
        10_000_000,
        401,
        0.01,
    );
}
