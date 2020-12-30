use criterion::{criterion_group, criterion_main, Criterion};
use etf::distributions::{Cauchy, CentralNormal, ChiSquared, Normal};
use etf::primitives::Distribution as _;
use rand::distributions::Distribution;
use rand_core::SeedableRng;
use rand_distr;
use rand_xoshiro::{Xoshiro128StarStar, Xoshiro256StarStar};

macro_rules! dist_benchmark_32 {
    ($group:ident, $etf_fn:ident, $rand_fn:ident, $etf_dist:expr, $rand_dist:expr) => {
        fn $etf_fn(c: &mut Criterion) {
            let dist = $etf_dist;
            let mut rng = Xoshiro128StarStar::seed_from_u64(0);
            c.bench_function(concat!(stringify!($group), "-etf"), |b| {
                b.iter(|| dist.sample(&mut rng))
            });
        }
        fn $rand_fn(c: &mut Criterion) {
            let dist = $rand_dist;
            let mut rng = Xoshiro128StarStar::seed_from_u64(0);
            c.bench_function(concat!(stringify!($group), "-rand"), |b| {
                b.iter(|| Distribution::<f32>::sample(&dist, &mut rng))
            });
        }

        criterion_group!($group, $rand_fn, $etf_fn);
    };
}

macro_rules! dist_benchmark_64 {
    ($group:ident, $etf_fn:ident, $rand_fn:ident, $etf_dist:expr, $rand_dist:expr) => {
        fn $etf_fn(c: &mut Criterion) {
            let dist = $etf_dist;
            let mut rng = Xoshiro256StarStar::seed_from_u64(0);
            c.bench_function(concat!(stringify!($group), "-etf"), |b| {
                b.iter(|| dist.sample(&mut rng))
            });
        }
        fn $rand_fn(c: &mut Criterion) {
            let dist = $rand_dist;
            let mut rng = Xoshiro256StarStar::seed_from_u64(0);
            c.bench_function(concat!(stringify!($group), "-rand"), |b| {
                b.iter(|| Distribution::<f64>::sample(&dist, &mut rng))
            });
        }

        criterion_group!($group, $rand_fn, $etf_fn);
    };
}

dist_benchmark_32!(
    central_normal_32,
    etf_central_normal_32_bench,
    rand_central_normal_32_bench,
    CentralNormal::new(1.0_f32).unwrap(),
    rand_distr::StandardNormal
);

dist_benchmark_64!(
    central_normal_64,
    etf_central_normal_64_bench,
    rand_central_normal_64_bench,
    CentralNormal::new(1.0_f64).unwrap(),
    rand_distr::StandardNormal
);

dist_benchmark_64!(
    normal_64,
    etf_normal_64_bench,
    rand_normal_64_bench,
    Normal::new(1.0_f64, 2.0_f64).unwrap(),
    rand_distr::Normal::new(1.0_f64, 2.0_f64).unwrap()
);

dist_benchmark_32!(
    cauchy_32,
    etf_cauchy_32_bench,
    rand_cauchy_32_bench,
    Cauchy::new(1.0_f32, 2.0_f32).unwrap(),
    rand_distr::Cauchy::new(1.0_f32, 2.0_f32).unwrap()
);

dist_benchmark_64!(
    cauchy_64,
    etf_cauchy_64_bench,
    rand_cauchy_64_bench,
    Cauchy::new(1.0_f64, 2.0_f64).unwrap(),
    rand_distr::Cauchy::new(1.0_f64, 2.0_f64).unwrap()
);

dist_benchmark_32!(
    chi_squared_32_k2,
    etf_chi_squared_32_k2_bench,
    rand_chi_squared_32_k2_bench,
    ChiSquared::new(2_f32).unwrap(),
    rand_distr::ChiSquared::new(2_f32).unwrap()
);

dist_benchmark_64!(
    chi_squared_64_k2,
    etf_chi_squared_64_k2_bench,
    rand_chi_squared_64_k2_bench,
    ChiSquared::new(2_f64).unwrap(),
    rand_distr::ChiSquared::new(2_f64).unwrap()
);

dist_benchmark_32!(
    chi_squared_32_k5,
    etf_chi_squared_32_k5_bench,
    rand_chi_squared_32_k5_bench,
    ChiSquared::new(5_f32).unwrap(),
    rand_distr::ChiSquared::new(5_f32).unwrap()
);

dist_benchmark_64!(
    chi_squared_64_k5,
    etf_chi_squared_64_k5_bench,
    rand_chi_squared_64_k5_bench,
    ChiSquared::new(5_f64).unwrap(),
    rand_distr::ChiSquared::new(5_f64).unwrap()
);

dist_benchmark_32!(
    chi_squared_32_k1000,
    etf_chi_squared_32_k1000_bench,
    rand_chi_squared_32_k1000_bench,
    ChiSquared::new(1000_f32).unwrap(),
    rand_distr::ChiSquared::new(1000_f32).unwrap()
);

dist_benchmark_64!(
    chi_squared_64_k1000,
    etf_chi_squared_64_k1000_bench,
    rand_chi_squared_64_k1000_bench,
    ChiSquared::new(1000_f64).unwrap(),
    rand_distr::ChiSquared::new(1000_f64).unwrap()
);

criterion_main!(
    central_normal_32,
    central_normal_64,
    normal_64,
    cauchy_32,
    cauchy_64,
    chi_squared_32_k2,
    chi_squared_64_k2,
    chi_squared_32_k5,
    chi_squared_64_k5,
    chi_squared_32_k1000,
    chi_squared_64_k1000,
);
