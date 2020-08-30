use super::util::{test_rng, TestFloat};
use std::collections::HashSet;
use rand::distributions::Distribution;

/// Returns the upper-tail P-value for the exact distribution of collision
/// events, based on Knuth' algorithm.
///
/// `k`: number of urns `n`: number of balls `c`: number of collisions
#[allow(dead_code)]
fn p_value(k: u64, n: u64, c: u64) -> f64 {
    let epsilon = 1e-20;
    let k_f64 = k as f64;
    let mut a = Vec::new();
    a.resize(1 + n as usize, 0f64);

    a[1] = 1.0;
    let mut j0 = 1;
    let mut j1 = 1;
    for _ in 1..n {
        j1 += 1;
        for j in (j0..=j1).rev() {
            let v = j as f64 / k_f64;
            a[j] = a[j] * v + a[j - 1] * (1.0 + 1.0 / k_f64 - v);
        }
        if a[j0] < epsilon {
            a[j0] = 0.0;
            j0 += 1;
        }
        if a[j1] < epsilon {
            a[j1] = 0.0;
            j1 -= 1;
        }
    }
    if (n - c) > j1 as u64 {
        return 1.0;
    }
    if (n - c) < j0 as u64 {
        return 0.0;
    }
    let mut cdf = 0.0;
    for j in ((n - c) as usize)..=j1 {
        cdf += a[j];
    }
    1.0 - cdf
}

/// Performs the Knuth collision test (1981).
///
/// The test simulates randomly throwing `n` balls into `k=2^d` urns, where `d`
/// is the dimension of the hypercube containing the urns.
///
/// The number of balls is computed based on the urn-to-ball ratio `k/n`. Knuth
/// originally suggested `k/n=64`, though other authors have used both greater
/// and lesser values.
///
/// The test is repeated several times, based on the `test_count` parameter.
///
#[allow(dead_code)]
pub fn collisions<T: TestFloat, D: Distribution<T>, F: Fn(f64) -> f64>(
    distribution: D,
    cdf: F,
    dimension: u8,
    urn_to_ball_ratio: u64,
    test_count: u64,
    p_value_threshold: f64,
) {
    let k = 1 << dimension;
    let n = k / urn_to_ball_ratio;
    let k_float = k as f64;
    // Associate a real number in [0, 1) to an urn.
    let find_urn = |r| ((r * k_float) as u64).min(k - 1);

    let mut p_value_sum = 0f64;
    let mut rng = test_rng();
    for _ in 0..test_count {
        let mut hash_set = HashSet::new();
        let mut collision_count = 0u64;
        for _ in 0..n {
            let r = cdf(distribution.sample(&mut rng).as_f64());
            let urn = find_urn(r);
            if !hash_set.insert(urn) {
                collision_count += 1;
            }
        }
        let p_value = p_value(k, n, collision_count);
        p_value_sum = p_value_sum + p_value;
    }

    let p_value = p_value_sum / test_count as f64;
    println!("Average P-value: {}", p_value);
    assert!(p_value > p_value_threshold);
}
