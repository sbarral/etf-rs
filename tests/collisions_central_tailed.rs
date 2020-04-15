mod common;
use common::*;

#[test]
fn collisions_central_tailed_f32() {
    collisions(
        central_normal_dist::<f32>(0.5, 2.0, 1e-4),
        |x| central_normal_cdf(x, 0.5),
        23,
        64,
        10,
        0.05,
    );
}

#[test]
fn collisions_central_tailed_f64() {
    collisions(
        central_normal_dist::<f64>(0.5, 2.0, 1e-4),
        |x| central_normal_cdf(x, 0.5),
        23,
        64,
        10,
        0.05,
    );
}
