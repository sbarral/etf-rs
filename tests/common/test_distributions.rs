
use super::TestFloat;
use etf::*;
use rand::distributions::Distribution;

// CDF for normal distribution.
#[allow(dead_code)]
pub fn central_normal_cdf<T: TestFloat>(x: T, variance: T) -> T {
    let two = T::ONE + T::ONE;
    (T::ONE + (x / (two * variance).sqrt()).erf()) / two
}

// Generic goodness of fit test for an ETF normal distribution.
#[allow(dead_code)]
pub fn central_normal_dist<T>(variance: T, tail_position: T, tolerance: T) -> impl Distribution<T>
where
    T: TestFloat,
    etf::partition::P128: etf::ValidSymmetricPartitionSize<T>,
{
    let two = T::ONE + T::ONE;
    let one_half = T::ONE / two;

    let pdf = move |x: T| (-one_half * x * x / variance).exp();
    let dpdf = move |x: T| -(x / variance) * (-one_half * x * x / variance).exp();

    // Build the distribution.
    type Partition = etf::partition::P128;

    let init_nodes =
        util::midpoint_prepartition::<Partition, _, _>(&pdf, T::ZERO, tail_position, 0);
    let table =
        util::newton_tabulation(&pdf, &dpdf, &init_nodes, &[], tolerance, T::ONE, 10).unwrap();
    let tail_func = util::WeibullEnvelope::new(
        one_half / tail_position,
        two,
        T::ONE,
        T::ZERO,
        tail_position,
        pdf,
    );
    DistCentralTailed::new(pdf, &table, tail_func, tail_func.area())
}
