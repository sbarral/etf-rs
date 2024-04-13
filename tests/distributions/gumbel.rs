use crate::common::{collisions, fair_goodness_of_fit};
use etf::distributions::Gumbel;
use std::f64;

// CDF for Gumbel distribution.
fn gumbel_cdf(x: f64, location: f64, scale: f64) -> f64 {
    let z = (x - location) / scale;
    f64::exp(-f64::exp(-z))
}

#[test]
fn gumbel_32_collisions() {
    let location = -1.7_f64;
    let scale = 2.8_f64;

    collisions(
        Gumbel::new(location as f32, scale as f32).unwrap(),
        |x| gumbel_cdf(x, location, scale),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn gumbel_64_collisions() {
    let location = -1.7_f64;
    let scale = 2.8_f64;

    collisions(
        Gumbel::new(location as f64, scale as f64).unwrap(),
        |x| gumbel_cdf(x, location, scale),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn gumbel_32_fit() {
    let location = 2.2_f64;
    let scale = 3.4_f64;

    fair_goodness_of_fit(
        Gumbel::new(location as f32, scale as f32).unwrap(),
        |x| gumbel_cdf(x, location, scale),
        50_000_000,
        401,
        0.01,
    );
}

#[test]

fn gumbel_64_fit() {
    let location = 2.2_f64;
    let scale = 3.4_f64;

    fair_goodness_of_fit(
        Gumbel::new(location as f64, scale as f64).unwrap(),
        |x| gumbel_cdf(x, location, scale),
        50_000_000,
        401,
        0.01,
    );
}
