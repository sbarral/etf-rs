mod common;
use common::*;

#[test]
fn fit_central_tailed_f32() {
    goodness_of_fit(
        central_normal_dist::<f32>(0.5, 2.0, 1e-4),
        |x| central_normal_cdf(x, 0.5),
        -3.0,
        3.0,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn fit_central_tailed_f64() {
    goodness_of_fit(
        central_normal_dist::<f64>(0.5, 2.0, 1e-4),
        |x| central_normal_cdf(x, 0.5),
        -3.0,
        3.0,
        10_000_000,
        401,
        0.01,
    );
}

#[test]
fn fit_rand_normal() {
    goodness_of_fit(
        rand_distr::Normal::new(0.0, 0.5_f64.sqrt()).unwrap(),
        |x| central_normal_cdf(x, 0.5),
        -3.0,
        3.0,
        10_000_000,
        401,
        0.01,
    );
}
