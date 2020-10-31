use crate::common::{collisions, goodness_of_fit};
use etf::distributions::Cauchy;
use etf::num::Float;

// CDF for normal distribution.
pub fn cauchy_cdf(x: f64, location: f64, scale: f64) -> f64 {
    ((x - location)/scale).atan() /f64::PI + 0.5
}

#[test]
fn cauchy_32_collisions() {
    let location = -1.7_f64;
    let scale = 2.8_f64;

    collisions(
        Cauchy::new(location as f32, scale as f32),
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
        Cauchy::new(location as f64, scale as f64),
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
    let n_scale = 4.0_f64; // test interval half-width in scale units

    goodness_of_fit(
        Cauchy::new(location as f32, scale as f32),
        |x| cauchy_cdf(x, location, scale),
        location - n_scale * scale,
        location + n_scale * scale,
        10_000_000,
        401,
        0.01,
    );
}

#[test]

fn cauchy_64_fit() {
    let location = 2.2_f64;
    let scale = 3.4_f64;
    let n_scale = 4.0_f64; // test interval half-width in scale units

    goodness_of_fit(
        Cauchy::new(location as f64, scale as f64),
        |x| cauchy_cdf(x, location, scale),
        location - n_scale * scale,
        location + n_scale * scale,
        10_000_000,
        401,
        0.01,
    );
}
