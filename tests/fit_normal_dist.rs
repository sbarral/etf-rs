mod common;
use common::{goodness_of_fit, TestFloat};
use etf::*;

// CDF (normalized).
fn normal_cdf<T: TestFloat>(x: T) -> T {
    let two = T::ONE + T::ONE;
    (T::ONE + x.erf()) / two
}
// PDF (may be scaled by constant factor).
fn normal_pdf<T: TestFloat>(x: T) -> T {
    (-x * x).exp()
}
// PDF derivative (scaling, if any, must be consistent with PDF).
fn normal_dpdf<T: TestFloat>(x: T) -> T {
    let two = T::ONE + T::ONE;
    -two * x * (-x * x).exp()
}

// Generic goodness of fit test for an ETF normal distribution.
fn fit_normal_dist<T>(x_tail: T, tolerance: T, x_max: T, n: u64, m: usize, p_value_threshold: T)
where
    T: TestFloat,
    etf::partition::P128: etf::ValidSymmetricPartitionSize<T>,
{
    // Build the distribution.
    let two = T::ONE + T::ONE;
    let one_half = T::ONE / two;
    type Partition = etf::partition::P128;

    let init_nodes =
        util::midpoint_prepartition::<Partition, _, _>(&normal_pdf, T::ZERO, x_tail, 0);
    let table = util::newton_tabulation(
        &normal_pdf,
        &normal_dpdf,
        &init_nodes,
        &[],
        tolerance,
        T::ONE,
        10,
    )
    .unwrap();
    let tail_func =
        util::WeibullEnvelope::new(one_half / x_tail, two, T::ONE, T::ZERO, x_tail, normal_pdf);
    let normal_dist = DistCentralTailed::new(normal_pdf, &table, tail_func, tail_func.area());

    goodness_of_fit(
        normal_dist,
        normal_cdf,
        -x_max,
        x_max,
        n,
        m,
        p_value_threshold,
    );
}

#[test]
fn fit_normal_dist_f32() {
    fit_normal_dist::<f32>(2.0, 1e-4, 3.0, 10_000_000, 401, 0.01);
}

#[test]
fn fit_normal_dist_f64() {
    fit_normal_dist::<f64>(2.0, 1e-4, 3.0, 10_000_000, 401, 0.01);
}
