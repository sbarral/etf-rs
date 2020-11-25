use crate::common::{collisions, fair_goodness_of_fit};
use etf::distributions::Cauchy;
use std::f64;

// CDF for Cauchy distribution.
fn cauchy_cdf(x: f64, location: f64, scale: f64) -> f64 {
    ((x - location)/scale).atan() /f64::consts::PI + 0.5
}

#[test]
fn cauchy_32_collisions() {
    let location = -1.7_f64;
    let scale = 2.8_f64;

    collisions(
        Cauchy::new(location as f32, scale as f32).unwrap(),
        |x| cauchy_cdf(x, location, scale),
        20,
        64,
        10,
        0.05,
    );
}


#[test]
fn cauchy_64_collisions() {
    let location = -1.7_f64;
    let scale = 2.8_f64;

    collisions(
        Cauchy::new(location as f64, scale as f64).unwrap(),
        |x| cauchy_cdf(x, location, scale),
        20,
        64,
        10,
        0.05,
    );
}

#[test]
fn cauchy_32_fit() {
    let location = 2.2_f64;
    let scale = 3.4_f64;
    
    fair_goodness_of_fit(
        Cauchy::new(location as f32, scale as f32).unwrap(),
        |x| cauchy_cdf(x, location, scale),
        50_000_000,
        401,
        0.01,
    );
}

#[test]

fn cauchy_64_fit() {
    let location = 2.2_f64;
    let scale = 3.4_f64;
    
    fair_goodness_of_fit(
        Cauchy::new(location as f64, scale as f64).unwrap(),
        |x| cauchy_cdf(x, location, scale),
        50_000_000,
        401,
        0.01,
    );
}
