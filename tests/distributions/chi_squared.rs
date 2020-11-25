use crate::common::{collisions, goodness_of_fit};
use special::Gamma;
use etf::distributions::ChiSquared;

fn chi_squared_cdf(x: f64, k: f64) -> f64 {
    (0.5*x).inc_gamma(0.5*k)
}

#[test]
fn chi_squared_32_collisions_k2() {
    let k = 2.0;

    collisions(
        ChiSquared::new(k as f32).unwrap(),
        |x| chi_squared_cdf(x, k),
        20,
        64,
        10,
        0.05,
    );
}


#[test]
fn chi_squared_64_collisions_k2() {
    let k = 2.0;

    collisions(
        ChiSquared::new(k as f64).unwrap(),
        |x| chi_squared_cdf(x, k),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn chi_squared_32_fit_k2() {
    let k = 2.0;
    let x_max = 25.0;

    goodness_of_fit(
        ChiSquared::new(k as f32).unwrap(),
        |x| chi_squared_cdf(x, k),
        0.0,
        x_max,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn chi_squared_64_fit_k2() {
    let k = 2.0;
    let x_max = 25.0;

    goodness_of_fit(
        ChiSquared::new(k as f64).unwrap(),
        |x| chi_squared_cdf(x, k),
        0.0,
        x_max,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn chi_squared_32_collisions_k4_5() {
    let k = 4.5;

    collisions(
        ChiSquared::new(k as f32).unwrap(),
        |x| chi_squared_cdf(x, k),
        20,
        64,
        10,
        0.05,
    );
}


#[test]
fn chi_squared_64_collisions_k4_5() {
    let k = 4.5;

    collisions(
        ChiSquared::new(k as f64).unwrap(),
        |x| chi_squared_cdf(x, k),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn chi_squared_32_fit_k4_5() {
    let k = 4.5;
    let x_max = 25.0;

    goodness_of_fit(
        ChiSquared::new(k as f32).unwrap(),
        |x| chi_squared_cdf(x, k),
        0.0,
        x_max,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn chi_squared_64_fit_k4_5() {
    let k = 4.5;
    let x_max = 25.0;

    goodness_of_fit(
        ChiSquared::new(k as f64).unwrap(),
        |x| chi_squared_cdf(x, k),
        0.0,
        x_max,
        10_000_000,
        401,
        0.01,
    );
}
