use criterion::{criterion_group, criterion_main, Criterion};
use etf::distributions::{Cauchy, CentralNormal, Normal};
use etf::primitives::Distribution as _;
use rand::distributions::Distribution;
use rand_distr;
//use rand_pcg;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

macro_rules! dist_benchmark {
    ($group:ident, $float:ty, $etf_fn:ident, $rand_fn:ident, $etf_dist:expr, $rand_dist:expr) => {
        fn $etf_fn(c: &mut Criterion) {
            let dist = $etf_dist;
            //let mut rng = rand_pcg::Lcg128Xsl64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
            let mut rng = Xoshiro256StarStar::seed_from_u64(0);
            c.bench_function(concat!(stringify!($group), "-etf"), |b| {
                b.iter(|| dist.sample(&mut rng))
            });
        }
        fn $rand_fn(c: &mut Criterion) {
            let dist = $rand_dist;
            //let mut rng = rand_pcg::Lcg128Xsl64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
            let mut rng = Xoshiro256StarStar::seed_from_u64(0);
            c.bench_function(concat!(stringify!($group), "-rand"), |b| {
                b.iter(|| Distribution::<$float>::sample(&dist, &mut rng))
            });
        }

        criterion_group!($group, $rand_fn, $etf_fn);
    };
}

dist_benchmark!(
    central_normal_64,
    f64,
    etf_central_normal_64_bench,
    rand_central_normal_64_bench,
    CentralNormal::new(1.0_f64),
    rand_distr::StandardNormal
);

dist_benchmark!(
    normal_64,
    f64,
    etf_normal_64_bench,
    rand_normal_64_bench,
    Normal::new(1.0_f64, 2.0_f64),
    rand_distr::Normal::new(1.0_f64, 2.0_f64).unwrap()
);

dist_benchmark!(
    cauchy_64,
    f64,
    etf_cauchy_64_bench,
    rand_cauchy_64_bench,
    Cauchy::new(1.0_f64, 2.0_f64),
    rand_distr::Cauchy::new(1.0_f64, 2.0_f64).unwrap()
);

criterion_main!(central_normal_64, normal_64, cauchy_64);
